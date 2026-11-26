package cn.langchat.core.runtime;

import cn.langchat.aigc.biz.entity.AigcAgent;
import cn.langchat.aigc.biz.entity.AigcKnowledge;
import cn.langchat.aigc.biz.entity.AigcMcp;
import cn.langchat.aigc.biz.entity.AigcModel;
import cn.langchat.aigc.biz.entity.AigcSkill;
import cn.langchat.aigc.biz.service.AigcAgentService;
import cn.langchat.aigc.biz.service.AigcKnowledgeService;
import cn.langchat.aigc.biz.service.AigcMcpService;
import cn.langchat.aigc.biz.service.AigcModelService;
import cn.langchat.aigc.biz.service.AigcSkillService;
import cn.langchat.common.exception.BizException;
import cn.langchat.core.runtime.model.AgentRuntimeDefinition;
import cn.langchat.core.support.CoreErrorCode;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * Agent 运行时配置加载器。
 *
 * <p>负责按 Agent ID 加载主体配置及其挂载的聊天模型、知识库、技能与 MCP，
 * 组装为 {@link AgentRuntimeDefinition} 供对话运行时消费。</p>
 *
 * @author LangChat Team
 * @since 2026/3/24
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AgentRuntimeLoader {

    /** JSON 数组 → List&lt;String&gt; 的类型引用，供 parseIdList 反序列化使用。 */
    private static final TypeReference<List<String>> STRING_LIST_TYPE = new TypeReference<>() {
    };

    private final AigcAgentService aigcAgentService;
    private final AigcKnowledgeService aigcKnowledgeService;
    private final AigcSkillService aigcSkillService;
    private final AigcMcpService aigcMcpService;
    private final AigcModelService aigcModelService;
    private final ObjectMapper objectMapper;

    /**
     * 按 Agent ID 加载完整运行时配置。
     *
     * <p>串联加载 Agent 挂载的四类资源：聊天模型、知识库、技能、MCP，
     * 最终组装为 {@link AgentRuntimeDefinition} 供对话运行时使用。</p>
     *
     * @param agentId Agent ID
     * @return 包含 Agent 主体与全部挂载资源的运行时定义
     */
    public AgentRuntimeDefinition load(String agentId) {
        // 加载 Agent 主体，不存在直接抛异常终止
        AigcAgent agent = aigcAgentService.getById(agentId);
        if (agent == null) {
            throw new BizException(CoreErrorCode.AGENT_NOT_FOUND);
        }

        // 依次加载聊天模型、知识库、技能与 MCP（各项内部各自负责校验与过滤）
        AigcModel chatModel = loadChatModel(agent);
        List<AigcKnowledge> knowledges = loadKnowledges(agent);
        List<AigcSkill> skills = loadSkills(agent);
        List<AigcMcp> mcps = loadMcps(agent);

        // 汇总加载结果日志，便于排查配置缺失问题
        log.info(
                "加载 Agent 运行时完成，agentId={}, chatModelId={}, knowledgeSize={}, skillSize={}, mcpSize={}",
                agentId,
                chatModel.getId(),
                knowledges.size(),
                skills.size(),
                mcps.size()
        );
        return new AgentRuntimeDefinition(agent, chatModel, knowledges, skills, mcps);
    }

    /**
     * 加载 Agent 绑定的聊天模型，未配置或模型不存在则抛 MODEL_NOT_FOUND。
     */
    private AigcModel loadChatModel(AigcAgent agent) {
        // 取推理模型 id；未配置时 firstNonBlank 返回 null，下方统一抛异常
        String modelId = firstNonBlank(agent.getReasoningModelId(), "");
        if (modelId == null) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND);
        }
        // 按模型 id 查询配置，不存在同样视为未配置
        AigcModel model = aigcModelService.getById(modelId);
        if (model == null) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND);
        }
        return model;
    }

    /**
     * 加载 Agent 挂载的知识库列表：解析 id 列表 → 批量查询 → 按配置顺序映射回实体，
     * 已被删除的知识库会被过滤跳过。
     */
    private List<AigcKnowledge> loadKnowledges(AigcAgent agent) {
        // 解析 Agent 存储的 knowledgeIds 字段（兼容 JSON 数组与分隔字符串两种格式）
        List<String> knowledgeIds = parseIdList(agent.getKnowledgeIds(), "knowledgeIds", agent.getId());
        if (knowledgeIds.isEmpty()) {
            return List.of();
        }
        // 批量查询知识库并按 id 建立映射；重复 id 仅保留首个
        Map<String, AigcKnowledge> knowledgeMap = aigcKnowledgeService.lambdaQuery()
                .in(AigcKnowledge::getId, knowledgeIds)
                .list()
                .stream()
                .collect(Collectors.toMap(AigcKnowledge::getId, Function.identity(), (left, right) -> left));
        // 按配置顺序将 id 映射回实体，查询不到（已被删除）的直接过滤跳过
        return knowledgeIds.stream()
                .map(knowledgeMap::get)
                .filter(Objects::nonNull)
                .toList();
    }

    /**
     * 加载 Agent 挂载的技能列表，只会返回处于启用状态（enabled=true）的技能，
     * 已被禁用或删除的技能会被过滤跳过。
     */
    private List<AigcSkill> loadSkills(AigcAgent agent) {
        // 解析 Agent 存储的 skillIds 字段
        List<String> skillIds = parseIdList(agent.getSkillIds(), "skillIds", agent.getId());
        if (skillIds.isEmpty()) {
            return List.of();
        }
        // 批量查询技能：仅取启用(enabled)的技能并按 id 建立映射（重复 id 保留首个）
        Map<String, AigcSkill> skillMap = aigcSkillService.lambdaQuery()
                .in(AigcSkill::getId, skillIds)
                .eq(AigcSkill::getEnabled, Boolean.TRUE)
                .list()
                .stream()
                .collect(Collectors.toMap(AigcSkill::getId, Function.identity(), (left, right) -> left));
        // 按配置顺序映射回实体；被禁用的技能未出现在 map 中，自然被过滤跳过
        return skillIds.stream()
                .map(skillMap::get)
                .filter(Objects::nonNull)
                .toList();
    }

    /**
     * 加载 Agent 挂载的 MCP 服务列表：解析 id 列表 → 批量查询 → 按配置顺序映射回实体，
     * 已被删除的 MCP 会被过滤跳过。
     */
    private List<AigcMcp> loadMcps(AigcAgent agent) {
        // 解析 Agent 存储的 mcpIds 字段
        List<String> mcpIds = parseIdList(agent.getMcpIds(), "mcpIds", agent.getId());
        if (mcpIds.isEmpty()) {
            return List.of();
        }
        // 批量查询 MCP 并按 id 建立映射；重复 id 仅保留首个
        Map<String, AigcMcp> mcpMap = aigcMcpService.lambdaQuery()
                .in(AigcMcp::getId, mcpIds)
                .list()
                .stream()
                .collect(Collectors.toMap(AigcMcp::getId, Function.identity(), (left, right) -> left));
        // 按配置顺序映射回实体，查询不到（已被删除）的直接过滤跳过
        return mcpIds.stream()
                .map(mcpMap::get)
                .filter(Objects::nonNull)
                .toList();
    }

    /**
     * 将 Agent 存储的原始字段解析为 id 列表：
     * 优先按 JSON 数组格式解析；解析失败或非数组时兜底按逗号/换行分隔字符串处理。
     * 两种路径均会做空值过滤与去重。
     */
    private List<String> parseIdList(String rawValue, String fieldName, String agentId) {
        // 字段为空直接返回空列表，无需进一步解析
        if (rawValue == null || rawValue.isBlank()) {
            return List.of();
        }
        String value = rawValue.trim();
        try {
            // 以 "[" 开头视为 JSON 数组格式，解析为字符串列表并去掉 null/空白项、去重
            if (value.startsWith("[")) {
                return objectMapper.readValue(value, STRING_LIST_TYPE).stream()
                        .filter(Objects::nonNull)
                        .map(String::trim)
                        .filter(item -> !item.isBlank())
                        .distinct()
                        .toList();
            }
        } catch (Exception ex) {
            // JSON 解析失败仅记录告警，继续走分隔字符串兜底逻辑
            log.warn("解析 Agent 字段失败，将按分隔字符串兜底处理，agentId={}, fieldName={}", agentId, fieldName, ex);
        }
        // 兜底：按逗号/换行拆分，剥离残留引号，过滤空白与重复项
        return List.of(value.split("[,\\n\\r]")).stream()
                .map(String::trim)
                .map(this::stripQuote)
                .filter(item -> !item.isBlank())
                .distinct()
                .toList();
    }

    /**
     * 剥离字符串两端可能残留的引号（从 JSON 字符串片段中拆分出的项常带引号），
     * 并去除首尾空白。
     */
    private String stripQuote(String value) {
        String normalized = value;
        if (normalized.startsWith("\"")) {
            normalized = normalized.substring(1);
        }
        if (normalized.endsWith("\"")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized.trim();
    }

    /**
     * 返回第一个非空白值；两者均空白则返回 null。
     */
    private String firstNonBlank(String first, String second) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        if (second != null && !second.isBlank()) {
            return second;
        }
        return null;
    }
}
