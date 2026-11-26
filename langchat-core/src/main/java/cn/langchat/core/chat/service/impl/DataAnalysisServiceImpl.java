package cn.langchat.core.chat.service.impl;

import cn.hutool.core.util.StrUtil;
import cn.langchat.aigc.biz.entity.AigcModel;
import cn.langchat.aigc.biz.service.AigcModelService;
import cn.langchat.common.exception.BizException;
import cn.langchat.core.chat.model.protocol.OpenAiChatCompletionChunk;
import cn.langchat.core.chat.model.request.DataAnalysisRequest;
import cn.langchat.core.chat.model.response.DataAnalysisCatalogItem;
import cn.langchat.core.chat.service.DataAnalysisService;
import cn.langchat.core.chat.support.AgentChatEventAssembler;
import cn.langchat.core.chat.support.AgentStreamFluxUtil;
import cn.langchat.core.chat.support.DataAnalysisChartBuilder;
import cn.langchat.core.runtime.DataAnalysisAiService;
import cn.langchat.core.runtime.factory.LangChain4jModelFactory;
import cn.langchat.core.support.CoreErrorCode;
import cn.langchat.datasource.entity.AigcDatasource;
import cn.langchat.datasource.model.DataQueryResult;
import cn.langchat.datasource.service.AigcDatasourceService;
import cn.langchat.monitor.model.ModelCallScene;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.tool.ToolExecutor;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;

/**
 * 智能问数运行时实现。
 *
 * <p>基于 LangChain4j AiServices 构建流式问数链路：向模型暴露 query_data（只读 SQL）、
 * inspect_selected_tables（查看表结构）、get_current_time（当前时间）三个工具，
 * 由模型自主决策 SQL 生成与执行，查询结果以 SSE 事件流返回表格与 ECharts 图表。</p>
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class DataAnalysisServiceImpl implements DataAnalysisService {

    /** 只读 SQL 查询工具名。 */
    private static final String QUERY_TOOL = "query_data";
    /** 查看授权表结构工具名。 */
    private static final String SCHEMA_TOOL = "inspect_selected_tables";
    /** 获取当前时间工具名。 */
    private static final String TIME_TOOL = "get_current_time";

    private final AigcDatasourceService datasourceService;
    private final AigcModelService modelService;
    private final LangChain4jModelFactory modelFactory;
    private final DataAnalysisChartBuilder chartBuilder;
    private final ObjectMapper objectMapper;

    /**
     * 列出可用于智能问数的数据源目录：仅取启用中的数据源，
     * 解析其表结构配置（只保留已导入且启用的表），组装为前端目录树。
     *
     * @return 数据源目录项列表（含数据源下的授权表信息）
     */
    @Override
    public List<DataAnalysisCatalogItem> listCatalog() {
        // 查询全部启用中的数据源，按更新时间倒序
        List<DataAnalysisCatalogItem> catalog = new ArrayList<>();
        List<AigcDatasource> datasources = datasourceService.lambdaQuery()
                .eq(AigcDatasource::getEnabled, true)
                .orderByDesc(AigcDatasource::getUpdateTime)
                .list();
        // 逐个解析表结构配置；无可用表的数据源跳过不展示
        for (AigcDatasource datasource : datasources) {
            List<ConfiguredTable> tables = parseConfiguredTables(datasource.getStructureJson());
            if (tables.isEmpty()) {
                continue;
            }
            catalog.add(new DataAnalysisCatalogItem(
                    datasource.getId(),
                    datasource.getName(),
                    datasource.getDbType(),
                    tables.stream().map(table -> new DataAnalysisCatalogItem.TableItem(
                            table.name(),
                            table.sourceName(),
                            table.comment(),
                            table.columns().size()
                    )).toList()
            ));
        }
        return catalog;
    }

    /**
     * 智能问数流式入口：校验并锁定授权表范围后，构建 SSE 事件流执行分析对话。
     * 会话 id 缺省时自动生成；事件由 AgentChatEventAssembler 统一组装。
     *
     * @param request 问数请求（数据源 id、模型 id、问题、所选表）
     * @return 兼容 OpenAI Chunk 格式的 SSE 事件流
     */
    @Override
    public Flux<ServerSentEvent<String>> stream(DataAnalysisRequest request) {
        // 解析授权表上下文，校验所选表均在授权范围内
        AnalysisContext context = resolveContext(request);
        AigcModel model = resolveModel(request.getModelId());
        // 会话 id 缺省时生成随机 id，供前端关联消息
        String conversationId = StrUtil.blankToDefault(
                request.getConversationId(),
                "analysis-" + UUID.randomUUID().toString().replace("-", "")
        );
        AgentChatEventAssembler assembler = new AgentChatEventAssembler(
                conversationId,
                "chatcmpl-" + UUID.randomUUID().toString().replace("-", ""),
                Instant.now().getEpochSecond(),
                model.getModel()
        );
        // 创建 SSE 事件流：异步执行分析，实时推送工具调用/表格/图表等事件
        return AgentStreamFluxUtil.createEventStream(
                sink -> runAnalysis(request, context, model, assembler, sink),
                assembler,
                objectMapper
        );
    }

    /**
     * 问数分析实际执行：构建流式模型 + 三工具的 AiService，推入授权表结构提示，
     * 处理流式回调（增量文本、工具调用前后、完成、异常），并在工具执行成功后推送表格与图表事件。
     * 通过 AtomicBoolean 保证完成/异常只触发一次收尾。
     */
    private void runAnalysis(
            DataAnalysisRequest request,
            AnalysisContext context,
            AigcModel model,
            AgentChatEventAssembler assembler,
            FluxSink<OpenAiChatCompletionChunk> sink
    ) {
        // 保证完成/异常收尾只执行一次
        AtomicBoolean completed = new AtomicBoolean(false);
        ConcurrentMap<String, QueryExecution> queryExecutions = new ConcurrentHashMap<>();
        try {
            // 构建流式聊天模型（场景 DATA_ANALYSIS，无 Agent 级参数覆盖）与三个工具
            StreamingChatModel chatModel = modelFactory.getStreamingChatModel(model, null, ModelCallScene.DATA_ANALYSIS);
            Map<ToolSpecification, ToolExecutor> tools = buildTools(context, queryExecutions);
            DataAnalysisAiService aiService = AiServices.builder(DataAnalysisAiService.class)
                    .streamingChatModel(chatModel)
                    .tools(tools)
                    .build();
            // 消息开始 + 推送授权表结构加载完成的状态提示
            sink.next(assembler.messageStart());
            sink.next(assembler.logDelta(
                    "已加载授权表结构",
                    "analysis.schema.loaded",
                    "completed",
                    Map.of("table_count", context.tables().size())
            ));
            // 启动流式分析：系统提示词约束 SQL 行为，用户问题驱动对话
            TokenStream stream = aiService.analyze(buildSystemPrompt(context), request.getQuestion());
            stream.onPartialResponse(partial -> sink.next(assembler.messageDelta(partial)))
                    // 工具调用前推送状态事件，帮助前端展示“正在查询”等中间态
                    .beforeToolExecution(before -> sink.next(assembler.beforeTool(before)))
                    // 工具执行后：推送工具结果事件；若为查询工具且成功，额外推送表格与图表
                    .onToolExecuted(execution -> {
                        sink.next(assembler.toolExecuted(execution));
                        QueryExecution query = queryExecutions.remove(execution.request().id());
                        if (!execution.hasFailed() && query != null) {
                            // 查询工具执行成功，推送表格数据与图表配置事件
                            emitAnalysisResult(query, assembler, sink);
                        }
                    })
                    .onCompleteResponse(response -> {
                        if (completed.compareAndSet(false, true)) {
                            completeStream(response, assembler, sink);
                        }
                    })
                    .onError(error -> {
                        if (completed.compareAndSet(false, true)) {
                            log.error("智能问数执行失败，datasourceId={}", request.getDatasourceId(), error);
                            sink.next(assembler.error(error));
                            sink.complete();
                        }
                    })
                    .start();
        } catch (Exception ex) {
            // 构建链路即失败（模型/工具构建异常），直接推送错误并结束流
            log.error("构建智能问数运行时失败，datasourceId={}", request.getDatasourceId(), ex);
            sink.next(assembler.error(ex));
            sink.complete();
        }
    }

    /**
     * 构建三个工具：query_data 执行只读 SQL 并按 request id 暂存结果、
     * inspect_selected_tables 返回授权表结构提示、get_current_time 返回当前时间。
     *
     * @param context    授权表上下文（含数据源与表白名单）
     * @param executions 工具调用 id → 查询执行结果的暂存表，供 onToolExecuted 推送事件
     * @return 工具规格与执行器映射
     */
    private Map<ToolSpecification, ToolExecutor> buildTools(
            AnalysisContext context,
            ConcurrentMap<String, QueryExecution> executions
    ) {
        Map<ToolSpecification, ToolExecutor> tools = new LinkedHashMap<>();
        ToolSpecification querySpec = ToolSpecification.builder()
                .name(QUERY_TOOL)
                .description("对用户已授权的表执行只读 SQL，并生成表格与可选图表")
                .parameters(JsonObjectSchema.builder()
                        .addStringProperty("sql", "必须是只读 SELECT/WITH 查询，使用真实表名和字段名")
                        .addStringProperty("title", "结果标题，使用简短中文")
                        .addStringProperty("chartType", "图表类型：auto、bar、line、pie、none")
                        .required(List.of("sql"))
                        .additionalProperties(false)
                        .build())
                .build();
        // 查询工具：解析 sql/title/chartType 入参，执行只读查询并暂存结果
        tools.put(querySpec, (toolRequest, memoryId) -> {
            Map<String, Object> arguments = parseToolArguments(toolRequest.arguments());
            String sql = stringValue(arguments.get("sql"));
            String title = stringValue(arguments.get("title"));
            String chartType = stringValue(arguments.get("chartType"));
            // 执行只读查询，SQL 只允许访问授权表；结果按工具调用 id 暂存，供事件推送
            DataQueryResult result = datasourceService.executeReadOnlyQuery(
                    context.datasource().getId(),
                    sql,
                    context.allowedTables()
            );
            executions.put(toolRequest.id(), new QueryExecution(sql, title, chartType, result));
            return writeJson(Map.of(
                    "columns", result.columns(),
                    "rowCount", result.rows().size(),
                    "rows", result.rows()
            ));
        });

        ToolSpecification schemaSpec = ToolSpecification.builder()
                .name(SCHEMA_TOOL)
                .description("重新查看当前已选择且允许 AI 检索的表结构")
                .parameters(JsonObjectSchema.builder().additionalProperties(false).build())
                .build();
        // 表结构工具：直接返回系统提示词中的授权表结构文本
        tools.put(schemaSpec, (toolRequest, memoryId) -> context.schemaPrompt());

        ToolSpecification timeSpec = ToolSpecification.builder()
                .name(TIME_TOOL)
                .description("获取当前系统时间，用于处理今天、本月、最近等相对时间问题")
                .parameters(JsonObjectSchema.builder().additionalProperties(false).build())
                .build();
        // 时间工具：返回当前系统时间，供模型解析“今天/本月/最近”等相对时间
        tools.put(timeSpec, (toolRequest, memoryId) -> OffsetDateTime.now().toString());
        return tools;
    }

    /**
     * 推送查询结果事件：先推表格事件（列/行数/行数据/SQL/标题），
     * 再按 chartType 用 ChartBuilder 生成 ECharts 配置并推送图表事件。
     */
    private void emitAnalysisResult(
            QueryExecution query,
            AgentChatEventAssembler assembler,
            FluxSink<OpenAiChatCompletionChunk> sink
    ) {
        // 组装表格事件载荷（完整行列数据 + SQL，供前端渲染表格）
        Map<String, Object> tablePayload = new LinkedHashMap<>();
        tablePayload.put("columns", query.result().columns());
        tablePayload.put("row_count", query.result().rows().size());
        tablePayload.put("rows", query.result().rows());
        tablePayload.put("sql", query.sql());
        tablePayload.put("title", defaultTitle(query.title()));
        sink.next(assembler.analysisTable(tablePayload));

        Map<String, Object> chartPayload = chartBuilder.build(
                query.chartType(),
                query.title(),
                query.result()
        );
        if (!chartPayload.isEmpty()) {
            // 图表配置非空时推送 ECharts 事件
            sink.next(assembler.analysisEchart(chartPayload));
        }
    }

    /** 收尾事件：推送 message_completed 与 message_stop（缺省 finishReason 取 stop），随后结束流。 */
    private void completeStream(
            ChatResponse response,
            AgentChatEventAssembler assembler,
            FluxSink<OpenAiChatCompletionChunk> sink
    ) {
        sink.next(assembler.messageCompleted(
                "analysis-message-" + UUID.randomUUID().toString().replace("-", ""),
                response
        ));
        sink.next(assembler.messageStop(
                response.finishReason() == null ? "stop" : response.finishReason().name().toLowerCase(Locale.ROOT)
        ));
        sink.complete();
    }

    /**
     * 解析问数上下文：校验请求非空（数据源/问题/所选表），检查数据源启用状态，
     * 比对所选表是否都在已导入且启用的表中（防越权），最终组装授权范围与表结构提示文本。
     */
    private AnalysisContext resolveContext(DataAnalysisRequest request) {
        // 请求基础校验：数据源 id、问题、表名列表缺一不可
        if (request == null || StrUtil.isBlank(request.getDatasourceId())
                || StrUtil.isBlank(request.getQuestion())
                || request.getTableNames() == null || request.getTableNames().isEmpty()) {
            throw new BizException(CoreErrorCode.INVALID_CHAT_REQUEST);
        }
        AigcDatasource datasource = datasourceService.getById(request.getDatasourceId());
        if (datasource == null || Boolean.FALSE.equals(datasource.getEnabled())) {
            throw new BizException(CoreErrorCode.INVALID_CHAT_REQUEST.code(), "数据源不存在或已停用");
        }
        // 收集请求所选表名（去空白、保序去重）
        Set<String> requested = request.getTableNames().stream()
                .filter(StrUtil::isNotBlank)
                .map(String::trim)
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
        // 从数据源表配置中筛出所选表；数量不一致说明有未导入/未启用的表，拒绝越权
        List<ConfiguredTable> selected = parseConfiguredTables(datasource.getStructureJson()).stream()
                .filter(table -> requested.contains(table.sourceName()))
                .toList();
        if (selected.isEmpty() || selected.size() != requested.size()) {
            throw new BizException(CoreErrorCode.INVALID_CHAT_REQUEST.code(), "选择中包含未导入或未启用的表");
        }
        // 授权表白名单 + 预生成表结构提示文本，供系统提示词与 SQL 约束使用
        Set<String> allowedTables = selected.stream()
                .map(ConfiguredTable::sourceName)
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
        return new AnalysisContext(datasource, selected, allowedTables, buildSchemaPrompt(selected));
    }

    /**
     * 解析问数使用的模型：指定 modelId 优先；未指定或模型不存在时
     * 回退到最早创建的 CHAT/REASONING 类型模型；仍无则抛 MODEL_NOT_FOUND。
     */
    private AigcModel resolveModel(String modelId) {
        AigcModel model = StrUtil.isNotBlank(modelId) ? modelService.getById(modelId) : null;
        if (model == null) {
            // 回退：取最早的 CHAT/REASONING 模型作为默认
            model = modelService.lambdaQuery()
                    .in(AigcModel::getType, List.of("CHAT", "REASONING"))
                    .orderByAsc(AigcModel::getCreateTime)
                    .last("LIMIT 1")
                    .one();
        }
        if (model == null) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND);
        }
        return model;
    }

    /**
     * 解析数据源的 structureJson 为表配置列表：仅取 imported=true 且 enabled=true 的表，
     * 把业务表名（name）与数据库源表名（sourceName）映射保留，列同样保留源名与 AI 名称。
     * 解析失败返回空列表不影响主流程（由调用处的 isEmpty 判断兜底）。
     */
    private List<ConfiguredTable> parseConfiguredTables(String structureJson) {
        if (StrUtil.isBlank(structureJson)) {
            return List.of();
        }
        try {
            JsonNode tablesNode = objectMapper.readTree(structureJson).path("tables");
            if (!tablesNode.isArray()) {
                return List.of();
            }
            List<ConfiguredTable> tables = new ArrayList<>();
            for (JsonNode tableNode : tablesNode) {
                if (!tableNode.path("imported").asBoolean(false)
                        || !tableNode.path("enabled").asBoolean(true)) {
                    continue;
                }
                String name = tableNode.path("tableName").asText("");
                String sourceName = tableNode.path("sourceTableName").asText(name);
                if (sourceName.isBlank()) {
                    continue;
                }
                List<ConfiguredColumn> columns = new ArrayList<>();
                for (JsonNode columnNode : tableNode.path("columns")) {
                    String aiName = columnNode.path("name").asText("");
                    String sourceColumn = columnNode.path("originalName").asText(aiName);
                    columns.add(new ConfiguredColumn(
                            sourceColumn,
                            aiName,
                            columnNode.path("type").asText(""),
                            columnNode.path("comment").asText("")
                    ));
                }
                tables.add(new ConfiguredTable(
                        name,
                        sourceName,
                        tableNode.path("tableComment").asText(""),
                        columns
                ));
            }
            return tables;
        } catch (Exception ex) {
            log.warn("解析智能问数表配置失败", ex);
            return List.of();
        }
    }

    /**
     * 构建系统提示词：约束模型只能用授权表、必须调 query_data 取数不得编造、
     * SQL 仅限 SELECT/WITH、控制结果规模、chartType 枚举、相对时间先取时间，
     * 末尾拼接授权表结构文本供模型生成合法 SQL。
     */
    private String buildSystemPrompt(AnalysisContext context) {
        return """
                你是企业数据分析助手。你只能基于当前授权表回答问题。
                必须遵守：
                1. 需要事实数据时必须调用 query_data，不得编造数字。
                2. SQL 只能是 SELECT 或 WITH，只能使用下方真实表名和真实字段名。
                3. 默认控制结果规模，明细查询不超过 200 行；优先聚合后再查询。
                4. query_data 的 chartType 只能使用 auto、bar、line、pie、none。
                5. 查询完成后用简洁中文总结关键结论，不要在正文重复完整表格或 ECharts JSON。
                6. 遇到“今天、本月、最近”等相对时间，先调用 get_current_time。

                授权表结构：
                """ + context.schemaPrompt();
    }

    /**
     * 拼接授权表结构提示文本：按“表名（业务名）：注释 + 列名 类型 业务名 含义”格式逐表输出，
     * 帮助模型使用源表名与字段名生成可执行 SQL。
     */
    private String buildSchemaPrompt(List<ConfiguredTable> tables) {
        StringBuilder prompt = new StringBuilder();
        for (ConfiguredTable table : tables) {
            prompt.append("\n表 ").append(table.sourceName());
            if (!table.name().equals(table.sourceName())) {
                prompt.append("（业务名称：").append(table.name()).append("）");
            }
            if (StrUtil.isNotBlank(table.comment())) {
                prompt.append("：").append(table.comment());
            }
            prompt.append('\n');
            for (ConfiguredColumn column : table.columns()) {
                prompt.append("- ").append(column.sourceName())
                        .append(" ").append(column.type());
                if (!column.aiName().equals(column.sourceName())) {
                    prompt.append("，业务字段名：").append(column.aiName());
                }
                if (StrUtil.isNotBlank(column.comment())) {
                    prompt.append("，含义：").append(column.comment());
                }
                prompt.append('\n');
            }
        }
        return prompt.toString();
    }

    /** 对象序列化为 JSON 字符串，失败抛 IllegalStateException 终止工具执行。 */
    private String writeJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("序列化查询结果失败", ex);
        }
    }

    /**
     * 解析工具入参 JSON 为有序 Map；入参为空或非法 JSON 时返回空 Map（等价于全部缺省处理）。
     */
    private Map<String, Object> parseToolArguments(String arguments) {
        try {
            JsonNode node = objectMapper.readTree(arguments == null ? "{}" : arguments);
            Map<String, Object> result = new LinkedHashMap<>();
            node.fields().forEachRemaining(entry ->
                    result.put(entry.getKey(), objectMapper.convertValue(entry.getValue(), Object.class)));
            return result;
        } catch (JsonProcessingException exception) {
            return Map.of();
        }
    }

    /** 值转字符串，null 回退空串。 */
    private String stringValue(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    /** 标题缺省回退“智能问数结果”。 */
    private String defaultTitle(String value) {
        return StrUtil.blankToDefault(value, "智能问数结果");
    }

    /** 本次问数上下文：数据源、授权表列表、允许表名白名单、表结构提示文本。 */
    private record AnalysisContext(
            AigcDatasource datasource,
            List<ConfiguredTable> tables,
            Set<String> allowedTables,
            String schemaPrompt
    ) {
    }

    /** 表配置：业务名、数据库源名、注释、列列表。 */
    private record ConfiguredTable(
            String name,
            String sourceName,
            String comment,
            List<ConfiguredColumn> columns
    ) {
    }

    /** 列配置：数据库源列名、AI 业务名、类型、注释。 */
    private record ConfiguredColumn(
            String sourceName,
            String aiName,
            String type,
            String comment
    ) {
    }

    /** 单次查询执行结果：SQL、标题、图表类型、查询返回数据，供事件推送缓存。 */
    private record QueryExecution(
            String sql,
            String title,
            String chartType,
            DataQueryResult result
    ) {
    }
}
