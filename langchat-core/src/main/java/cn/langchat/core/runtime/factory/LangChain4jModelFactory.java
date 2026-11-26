package cn.langchat.core.runtime.factory;

import cn.hutool.core.util.StrUtil;
import cn.langchat.aigc.biz.entity.AigcAgent;
import cn.langchat.aigc.biz.entity.AigcModel;
import cn.langchat.aigc.biz.support.ModelConfigSupport;
import cn.langchat.common.ai.enums.AiProviderType;
import cn.langchat.common.exception.BizException;
import cn.langchat.core.support.CoreErrorCode;
import cn.langchat.monitor.listener.ModelCallListenerFactory;
import cn.langchat.monitor.model.ModelCallMetadata;
import cn.langchat.monitor.model.ModelCallScene;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.community.model.dashscope.QwenEmbeddingModel;
import dev.langchain4j.community.model.dashscope.QwenStreamingChatModel;
import dev.langchain4j.community.model.zhipu.ZhipuAiEmbeddingModel;
import dev.langchain4j.community.model.zhipu.ZhipuAiStreamingChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiStreamingChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * LangChain4j 模型工厂。
 *
 * <p>构建模型客户端时统一注入模型调用监听器（{@code langchat-monitor} 模块提供），
 * 使每一次模型调用都能被监控模块采集。向量模型部分供应商的构建器不支持监听器，
 * 统一通过 {@code EmbeddingModel.addListener(...)} 包装注入。</p>
 *
 * @author LangChat Team
 * @since 2026/3/24
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class LangChain4jModelFactory {

    /** 本地/私化模型（如 Ollama）无需 API Key 时使用的非空占位值，避免构建器校验报错。 */
    private static final String EMPTY_API_KEY = "EMPTY_API_KEY";
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final ModelCallListenerFactory modelCallListenerFactory;

    private final ConcurrentMap<String, StreamingChatModel> chatModelCache = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, EmbeddingModel> embeddingModelCache = new ConcurrentHashMap<>();

    /**
     * 构建流式聊天模型。
     *
     * @param model 模型配置
     * @param agent Agent 配置，可为空
     * @param scene 调用场景，用于监控报表按业务链路区分消耗
     */
    public StreamingChatModel getStreamingChatModel(AigcModel model, AigcAgent agent, String scene) {
        // 解析 Agent 级模型参数覆盖（温度/topP/maxTokens），agent 为空或解析失败时回退为空配置
        AgentModelConfig config = parseAgentModelConfig(agent);
        // 缓存键 = 模型 id + 更新时间 + Agent 参数覆盖 + 调用场景，任一变化即命中新键触发重建
        String cacheKey = String.join(
                ":",
                model.getId(),
                String.valueOf(model.getUpdateTime()),
                String.valueOf(temperature(config, model)),
                String.valueOf(topP(config, model)),
                String.valueOf(maxTokens(config, model)),
                String.valueOf(scene)
        );
        // 命中缓存直接复用，缺失时按供应商创建（含监控监听器注入）并写入缓存
        return chatModelCache.computeIfAbsent(cacheKey, key -> createStreamingChatModel(model, config, scene));
    }

    /**
     * 构建流式聊天模型（默认 Agent 对话场景）。
     */
    public StreamingChatModel getStreamingChatModel(AigcModel model, AigcAgent agent) {
        return getStreamingChatModel(model, agent, ModelCallScene.AGENT_CHAT);
    }

    /**
     * 构建向量模型。
     *
     * @param model 模型配置
     * @param scene 调用场景，用于监控报表按业务链路区分消耗
     */
    public EmbeddingModel getEmbeddingModel(AigcModel model, String scene) {
        String cacheKey = model.getId() + ":" + model.getUpdateTime() + ":" + scene;
        return embeddingModelCache.computeIfAbsent(cacheKey, key -> createEmbeddingModel(model, scene));
    }

    /**
     * 构建向量模型（默认知识库向量化场景）。
     */
    public EmbeddingModel getEmbeddingModel(AigcModel model) {
        return getEmbeddingModel(model, ModelCallScene.KNOWLEDGE_INDEX);
    }

    /**
     * 按供应商创建流式聊天模型实例（仅缓存未命中时调用），
     * 统一注入监控监听器，再按 provider 路由到各供应商专属构建方法，
     * 由它们处理 SDK 间参数命名、端点处理等差异。
     */
    private StreamingChatModel createStreamingChatModel(AigcModel model, AgentModelConfig config, String scene) {
        AiProviderType provider = AiProviderType.fromCode(model.getProvider());
        log.info("构建流式聊天模型，modelId={}, provider={}, model={}, scene={}",
                model.getId(), provider, model.getModel(), scene);
        // 注入模型调用监听器，使本次对话的 token 消耗与耗时可被监控模块采集
        List<ChatModelListener> listeners = List.of(modelCallListenerFactory.createChatListener(metadataOf(model, scene)));
        // 按供应商路由：OpenAI 系走通用兼容实现，其余供应商各有独立构建方法
        return switch (provider) {
            case OPENAI, DEEPSEEK, OPENAI_COMPATIBLE -> openAiCompatibleStreamingModel(model, config, listeners);
            case OLLAMA -> ollamaStreamingModel(model, config, listeners);
            case DASHSCOPE -> dashScopeStreamingModel(model, config, listeners);
            case ZHIPU -> zhipuStreamingModel(model, config, listeners);
            case GEMINI -> geminiStreamingModel(model, config, listeners);
            default -> throw new BizException(CoreErrorCode.UNSUPPORTED_PROVIDER);
        };
    }

    /**
     * 按供应商创建向量模型实例（仅缓存未命中时调用）：注入监控监听器后按 provider 路由；
     * DashScope / 智谱构建器不支持 listeners，统一在创建后用 addListener 补齐。
     */
    private EmbeddingModel createEmbeddingModel(AigcModel model, String scene) {
        AiProviderType provider = AiProviderType.fromCode(model.getProvider());
        log.info("构建向量模型，modelId={}, provider={}, model={}, scene={}",
                model.getId(), provider, model.getModel(), scene);
        EmbeddingModelListener listener = modelCallListenerFactory.createEmbeddingListener(metadataOf(model, scene));
        EmbeddingModel embeddingModel = switch (provider) {
            case OPENAI, DEEPSEEK, OPENAI_COMPATIBLE -> openAiCompatibleEmbeddingModel(model, List.of(listener));
            case OLLAMA -> ollamaEmbeddingModel(model, List.of(listener));
            case DASHSCOPE -> dashScopeEmbeddingModel(model);
            case ZHIPU -> zhipuEmbeddingModel(model);
            case GEMINI -> geminiEmbeddingModel(model, List.of(listener));
            default -> throw new BizException(CoreErrorCode.UNSUPPORTED_PROVIDER);
        };
        // DashScope / 智谱 的构建器不提供 listeners(...) 方法，用官方包装器补齐；
        // addListener 对已注册该监听器的模型会识别并叠加，不会重复触发。
        return embeddingModel.addListener(listener);
    }

    /** 组装模型调用元数据（模型 id/名称/供应商/场景），供监控监听器按维度上报。 */
    private ModelCallMetadata metadataOf(AigcModel model, String scene) {
        return ModelCallMetadata.of(model.getId(), model.getModel(), model.getProvider(), scene);
    }

    /**
     * 构建 OpenAI 兼容流式模型，统一覆盖 OpenAI / DeepSeek / 自定义 OpenAI 兼容服务，
     * 均走标准 OpenAI Chat Completions 协议。
     */
    private StreamingChatModel openAiCompatibleStreamingModel(
            AigcModel model, AgentModelConfig config, List<ChatModelListener> listeners) {
        OpenAiStreamingChatModel.OpenAiStreamingChatModelBuilder builder = OpenAiStreamingChatModel.builder()
                .modelName(model.getModel())
                .apiKey(defaultApiKey(model))
                .temperature(temperature(config, model))
                .topP(topP(config, model))
                .maxTokens(maxTokens(config, model))
                .timeout(timeout(model))
                .listeners(listeners);
        applyBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /**
     * 构建 Ollama 流式模型（本地或私化部署，无需 API Key）。
     * 注意 Ollama 的最大生成长度参数名为 numPredict，对应 OpenAI 的 maxTokens。
     */
    private StreamingChatModel ollamaStreamingModel(
            AigcModel model, AgentModelConfig config, List<ChatModelListener> listeners) {
        OllamaStreamingChatModel.OllamaStreamingChatModelBuilder builder = OllamaStreamingChatModel.builder()
                .modelName(model.getModel())
                .temperature(temperature(config, model))
                .topP(topP(config, model))
                // numPredict 即最大生成长度（Ollama 命名与 OpenAI 的 maxTokens 不同）
                .numPredict(maxTokens(config, model))
                .timeout(timeout(model))
                .listeners(listeners);
        applyBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /**
     * 构建 DashScope（通义千问）原生流式模型：温度参数需转 Float 适配 SDK，
     * 端点用专用过滤逻辑（见 applyDashScopeBaseUrl），不能透传 OpenAI 兼容地址。
     */
    private StreamingChatModel dashScopeStreamingModel(
            AigcModel model, AgentModelConfig config, List<ChatModelListener> listeners) {
        QwenStreamingChatModel.QwenStreamingChatModelBuilder builder = QwenStreamingChatModel.builder()
                .modelName(model.getModel())
                .apiKey(defaultApiKey(model))
                // Qwen SDK 的温度参数类型为 Float 而非 Double
                .temperature(toFloat(temperature(config, model)))
                .topP(topP(config, model))
                .maxTokens(maxTokens(config, model))
                .listeners(listeners);
        // DashScope 原生 SDK 自带默认端点，不能把 OpenAI 兼容地址（compatible-mode）传给它
        applyDashScopeBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /**
     * 构建 OpenAI 兼容向量模型，覆盖 OpenAI / DeepSeek / 自定义 OpenAI 兼容服务，
     * 走标准 OpenAI Embeddings 协议。
     */
    private EmbeddingModel openAiCompatibleEmbeddingModel(AigcModel model, List<EmbeddingModelListener> listeners) {
        OpenAiEmbeddingModel.OpenAiEmbeddingModelBuilder builder = OpenAiEmbeddingModel.builder()
                .modelName(model.getModel())
                .apiKey(defaultApiKey(model))
                .dimensions(embeddingDimension(model))
                .timeout(timeout(model))
                .listeners(listeners);
        applyBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /** 构建 Ollama 向量模型（本地部署，无需 API Key）。 */
    private EmbeddingModel ollamaEmbeddingModel(AigcModel model, List<EmbeddingModelListener> listeners) {
        OllamaEmbeddingModel.OllamaEmbeddingModelBuilder builder = OllamaEmbeddingModel.builder()
                .modelName(model.getModel())
                .timeout(timeout(model))
                .listeners(listeners);
        applyBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /**
     * 构建 DashScope 向量模型：需指定向量维度；端点过滤兼容地址避免原生 SDK 404。
     */
    private EmbeddingModel dashScopeEmbeddingModel(AigcModel model) {
        QwenEmbeddingModel.QwenEmbeddingModelBuilder builder = QwenEmbeddingModel.builder()
                .modelName(model.getModel())
                .apiKey(defaultApiKey(model))
                .dimension(embeddingDimension(model));
        // 同上：过滤 OpenAI 兼容地址，避免原生 SDK 404
        applyDashScopeBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /**
     * 构建智谱（GLM）流式模型；注意模型名参数为 model 而非 modelName，
     * 最大生成长度参数名为 maxToken（无 s）。
     */
    private StreamingChatModel zhipuStreamingModel(
            AigcModel model, AgentModelConfig config, List<ChatModelListener> listeners) {
        ZhipuAiStreamingChatModel.ZhipuAiStreamingChatModelBuilder builder =
                ZhipuAiStreamingChatModel.builder()
                        .model(model.getModel())
                        .apiKey(defaultApiKey(model))
                        .temperature(temperature(config, model))
                        .topP(topP(config, model))
                        // 智谱 SDK 的生成长度参数名为 maxToken，与 OpenAI 的 maxTokens 命名不同
                        .maxToken(maxTokens(config, model))
                        .listeners(listeners);
        applyBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /** 构建智谱向量模型；智谱构建器不提供 listeners(...)，由调用方 addListener 补齐。 */
    private EmbeddingModel zhipuEmbeddingModel(AigcModel model) {
        ZhipuAiEmbeddingModel.ZhipuAiEmbeddingModelBuilder builder = ZhipuAiEmbeddingModel.builder()
                .model(model.getModel())
                .apiKey(defaultApiKey(model));
        applyBaseUrl(model, builder::baseUrl);
        return builder.build();
    }

    /**
     * 构建 Google Gemini 流式模型；端点固定官方地址，不支持自定义 baseUrl，
     * 生成长度参数名为 maxOutputTokens。
     */
    private StreamingChatModel geminiStreamingModel(
            AigcModel model, AgentModelConfig config, List<ChatModelListener> listeners) {
        return GoogleAiGeminiStreamingChatModel.builder()
                .apiKey(defaultApiKey(model))
                .modelName(model.getModel())
                .temperature(temperature(config, model))
                .topP(topP(config, model))
                .maxOutputTokens(maxTokens(config, model))
                .listeners(listeners)
                .build();
    }

    /** 构建 Google Gemini 向量模型；端点固定官方地址，不支持自定义 baseUrl。 */
    private EmbeddingModel geminiEmbeddingModel(AigcModel model, List<EmbeddingModelListener> listeners) {
        return GoogleAiEmbeddingModel.builder()
                .apiKey(defaultApiKey(model))
                .modelName(model.getModel())
                .listeners(listeners)
                .build();
    }

    /** 解析请求超时时间；未配置或非正值时默认 3 分钟。 */
    private Duration timeout(AigcModel model) {
        int minutes = model.getTimeout() == null || model.getTimeout() <= 0 ? 3 : model.getTimeout();
        return Duration.ofMinutes(minutes);
    }

    /**
     * 解析向量维度，必须配置为正整数，否则抛出业务异常——
     * 向量库建表（集合）依赖维度，与实际模型维度不一致会导致写入失败。
     */
    private Integer embeddingDimension(AigcModel model) {
        Integer dimension = ModelConfigSupport.resolveDimension(model);
        if (dimension == null) {
            throw new BizException(
                    CoreErrorCode.INVALID_MODEL_CONFIG.code(),
                    "向量模型 configJson.dimension 必须配置为大于 0 的整数"
            );
        }
        return dimension;
    }

    /** 最大生成长度：优先 Agent 级覆盖，未覆盖则沿用模型库配置。 */
    private Integer maxTokens(AgentModelConfig config, AigcModel model) {
        return firstNonNull(config.maxOutputTokens(), model.getMaxToken());
    }

    /** 采样温度：Agent 级覆盖 → 模型库配置 → 默认 0.7。 */
    private Double temperature(AgentModelConfig config, AigcModel model) {
        return firstNonNull(config.temperature(), model.getTemperature(), 0.7D);
    }

    /** 核采样概率：Agent 级覆盖 → 模型库配置 → 默认 0.9。 */
    private Double topP(AgentModelConfig config, AigcModel model) {
        return firstNonNull(config.topP(), model.getTopP(), 0.9D);
    }

    /**
     * 解析 Agent 的模型配置 JSON；为空或解析失败时回退空配置（空配置内各参数为 null，
     * 由 temperature/topP/maxTokens 的默认值兜底）。
     */
    private AgentModelConfig parseAgentModelConfig(AigcAgent agent) {
        if (agent == null || StrUtil.isBlank(agent.getModelConfigJson())) {
            return AgentModelConfig.empty();
        }
        try {
            return OBJECT_MAPPER.readValue(agent.getModelConfigJson(), AgentModelConfig.class);
        } catch (Exception ex) {
            log.warn("解析 Agent 模型配置失败，agentId={}", agent.getId(), ex);
            return AgentModelConfig.empty();
        }
    }

    /** 未配置 API Key 时返回占位值，避免部分构建器对空 Key 的校验报错（本地模型无需真实凭证）。 */
    private String defaultApiKey(AigcModel model) {
        return StrUtil.isBlank(model.getApiKey()) ? EMPTY_API_KEY : model.getApiKey();
    }

    /**
     * 将模型库的 baseUrl 透传给构建器：可指向 DeepSeek、OpenAI 兼容网关或任意私化服务；
     * 为空时由各 SDK 使用自己的默认端点。
     */
    private void applyBaseUrl(AigcModel model, java.util.function.Consumer<String> consumer) {
        consumer.accept(model.getBaseUrl());
    }
    /**
     * DashScope 原生 SDK 端点处理。
     *
     * <p>实测结论：官方 {@code dashscope.aliyuncs.com} 的地址（无论是裸域名、
     * {@code /api/v1} 还是 OpenAI 兼容地址）一旦显式传给原生 SDK，都会因为
     * SDK 自身的路径拼接规则导致 404，因此官方域名一律忽略、使用 SDK 默认端点；
     * 只有私化部署（非 aliyuncs.com 域名）才按裸域名覆盖。
     */
    private void applyDashScopeBaseUrl(AigcModel model, java.util.function.Consumer<String> consumer) {
        String baseUrl = model.getBaseUrl();
        if (StrUtil.isBlank(baseUrl) || baseUrl.contains("aliyuncs.com")) {
            return;
        }
        try {
            java.net.URI uri = java.net.URI.create(baseUrl.trim());
            if (uri.getHost() == null) {
                return;
            }
            String host = uri.getScheme()
                    + "://"
                    + uri.getHost()
                    + (uri.getPort() > 0 ? ":" + uri.getPort() : "");
            log.info("DashScope 使用私化部署端点：{}", host);
            consumer.accept(host);
        } catch (Exception ex) {
            log.warn("忽略无法解析的 DashScope baseUrl={}，使用官方默认端点", baseUrl, ex);
        }
    }

    /** 优先级链：返回第一个非空值，用于「Agent 覆盖 → 模型库默认 → 内置兜底」三级取值。 */
    @SafeVarargs
    private <T> T firstNonNull(T... values) {
        for (T value : values) {
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    /** 适配 DashScope（Qwen）SDK：温度参数类型为 Float。 */
    private Float toFloat(Double value) {
        return value == null ? null : value.floatValue();
    }

    /** Agent 级模型参数覆盖项（解析自 modelConfigJson），null 表示不覆盖、沿用模型库配置。 */
    private record AgentModelConfig(
            Double temperature,
            Double topP,
            Integer maxOutputTokens
    ) {
        /** 全 null 的空配置，等价于完全沿用模型库默认值。 */
        private static AgentModelConfig empty() {
            return new AgentModelConfig(null, null, null);
        }
    }
}
