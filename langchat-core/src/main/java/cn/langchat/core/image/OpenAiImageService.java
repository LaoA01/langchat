package cn.langchat.core.image;

import cn.hutool.core.util.StrUtil;
import cn.langchat.aigc.biz.entity.AigcModel;
import cn.langchat.aigc.biz.service.AigcModelService;
import cn.langchat.common.ai.enums.AiProviderType;
import cn.langchat.common.ai.model.ImageGenerationRequest;
import cn.langchat.common.ai.model.ImageGenerationResult;
import cn.langchat.common.ai.service.ImageAiService;
import cn.langchat.common.exception.BizException;
import cn.langchat.core.support.CoreErrorCode;
import cn.langchat.monitor.model.ModelCallMetadata;
import cn.langchat.monitor.model.ModelCallRecord;
import cn.langchat.monitor.model.ModelCallScene;
import cn.langchat.monitor.model.ModelCallType;
import cn.langchat.monitor.service.ModelCallRecorder;
import com.fasterxml.jackson.databind.JsonNode;
import dev.langchain4j.community.model.dashscope.WanxImageModel;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiImageModel;
import dev.langchain4j.model.output.Response;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

/**
 * OpenAI 兼容协议的图片 AI 实现。
 *
 * <p>按 modelId 从模型管理读取模型配置并构造底层模型客户端，
 * 图片生成使用 {@link OpenAiImageModel}，图片识别 (OCR) 使用
 * {@link OpenAiChatModel} 的多模态能力。</p>
 *
 * @author LangChat Team
 * @since 2026/8/27
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class OpenAiImageService implements ImageAiService {

    /** 未配置 API Key 时使用的非空占位值，避免模型构建器的非空校验失败。 */
    private static final String EMPTY_API_KEY = "EMPTY_API_KEY";
    /** 默认生成图片尺寸（请求未指定 size 时使用）。 */
    private static final String DEFAULT_IMAGE_SIZE = "1024x1024";
    /** 默认生成图片质量（请求未指定 quality 时使用）。 */
    private static final String DEFAULT_IMAGE_QUALITY = "standard";
    /** 默认生成图片张数（请求未指定 n 或 n 非正数时使用）。 */
    private static final int DEFAULT_IMAGE_COUNT = 1;
    /** OCR 默认提示词（调用方未传入提示词时使用）。 */
    private static final String DEFAULT_OCR_PROMPT = "请识别图片中的文字内容并原样输出。";

    private final AigcModelService aigcModelService;
    private final ModelCallRecorder modelCallRecorder;

    /**
     * 文生图入口：按 modelId 加载模型配置，依据供应商类型路由到对应实现，
     * 并统一记录监控指标（场景 IMAGE_GENERATE / 类型 IMAGE）。
     *
     * @param modelId 模型配置 id（对应模型管理中的记录）
     * @param request 生成请求（提示词、尺寸、质量、张数等）
     * @return 生成结果（取第一张图的 url / base64 / 修正提示词）
     */
    @Override
    public ImageGenerationResult generateImage(String modelId, ImageGenerationRequest request) {
        // 加载模型配置并解析供应商类型，决定走哪条生成链路
        AigcModel model = loadModel(modelId);
        AiProviderType provider = AiProviderType.fromCode(model.getProvider());
        log.info("生成图片，modelId={}, provider={}, model={}, size={}, n={}",
                modelId, provider, model.getModel(), request.size(), request.n());
        long startTime = System.currentTimeMillis();
        try {
            // 按供应商路由：Gemini 走原生 :predict 协议，DashScope 走通义万相，其余走 OpenAI 兼容协议
            ImageGenerationResult result = switch (provider) {
                case GEMINI -> geminiGenerateImage(model, request);
                case DASHSCOPE -> wanxGenerateImage(model, request);
                default -> openAiGenerateImage(model, request);
            };
            // 成功则上报监控记录（张数按实际请求计，缺省为 1）
            modelCallRecorder.record(ModelCallRecord.builder()
                    .metadata(metadataOf(model, ModelCallScene.IMAGE_GENERATE))
                    .callType(ModelCallType.IMAGE)
                    .success(true)
                    .itemCount(resolveImageCount(request))
                    .duration(System.currentTimeMillis() - startTime)
                    .build());
            return result;
        } catch (RuntimeException ex) {
            // 失败同样上报监控（携带异常信息），随后原样抛出由上层处理
            modelCallRecorder.record(ModelCallRecord.builder()
                    .metadata(metadataOf(model, ModelCallScene.IMAGE_GENERATE))
                    .callType(ModelCallType.IMAGE)
                    .success(false)
                    .duration(System.currentTimeMillis() - startTime)
                    .errorMessage(ex.getMessage())
                    .build());
            throw ex;
        }
    }

    /** 组装模型调用元数据（模型 id/名称/供应商/场景），供监控记录按维度上报。 */
    private ModelCallMetadata metadataOf(AigcModel model, String scene) {
        return ModelCallMetadata.of(model.getId(), model.getModel(), model.getProvider(), scene);
    }

    /** 解析实际生成张数：请求未指定 n 或 n 非正数时回退为默认 1 张。 */
    private Integer resolveImageCount(ImageGenerationRequest request) {
        return request.n() == null || request.n() <= 0 ? DEFAULT_IMAGE_COUNT : request.n();
    }

    /**
     * OpenAI 系文生图（含各类 OpenAI 兼容服务），走标准 OpenAI Images 协议。
     * 尺寸/质量未指定时使用默认值（1024x1024 / standard），仅取返回的第一张图。
     */
    private ImageGenerationResult openAiGenerateImage(AigcModel model, ImageGenerationRequest request) {
        // 构建图像模型客户端：尺寸/质量缺省回退默认值，超时取模型配置（默认 3 分钟）
        OpenAiImageModel.OpenAiImageModelBuilder builder = OpenAiImageModel.builder()
                .modelName(model.getModel())
                .apiKey(defaultApiKey(model))
                .size(StrUtil.blankToDefault(request.size(), DEFAULT_IMAGE_SIZE))
                .quality(StrUtil.blankToDefault(request.quality(), DEFAULT_IMAGE_QUALITY))
                .timeout(timeout(model));
        // 模型配置了自定义 baseUrl（如代理或兼容网关）时覆盖默认 OpenAI 端点
        applyBaseUrl(model, builder::baseUrl);
        OpenAiImageModel imageModel = builder.build();

        // 解析生成张数（缺省为 1），调用 Images 接口生成
        int count = request.n() == null || request.n() <= 0 ? DEFAULT_IMAGE_COUNT : request.n();
        Response<List<Image>> response = imageModel.generate(request.prompt(), count);
        // 返回为空视为生成失败，统一抛业务异常
        if (response == null || response.content() == null || response.content().isEmpty()) {
            throw new BizException(CoreErrorCode.CHAT_RUNTIME_ERROR);
        }

        // 仅取第一张图，url 可能为空（部分模型仅返回 base64）
        Image image = response.content().get(0);
        String url = image.url() == null ? null : image.url().toString();
        return new ImageGenerationResult(url, image.base64Data(), image.revisedPrompt());
    }

    /**
     * 通义万相文生图（DashScope 官方协议）。
     */
    private ImageGenerationResult wanxGenerateImage(AigcModel model, ImageGenerationRequest request) {
        // 通义万相按 DashScope 官方协议调用，仅需模型名与 API Key
        int count = request.n() == null || request.n() <= 0 ? DEFAULT_IMAGE_COUNT : request.n();
        WanxImageModel imageModel = WanxImageModel.builder()
                .modelName(model.getModel())
                .apiKey(defaultApiKey(model))
                .build();
        // 调用通义万相生成，返回为空则视为失败抛业务异常
        Response<List<Image>> response = imageModel.generate(request.prompt(), count);
        if (response == null || response.content() == null || response.content().isEmpty()) {
            throw new BizException(CoreErrorCode.CHAT_RUNTIME_ERROR);
        }
        // 仅取第一张图，url 由 DashScope 托管
        Image image = response.content().get(0);
        String url = image.url() == null ? null : image.url().toString();
        return new ImageGenerationResult(url, image.base64Data(), image.revisedPrompt());
    }

    /**
     * Gemini 文生图（generativelanguage :predict 标准协议，返回 base64）。
     */
    private ImageGenerationResult geminiGenerateImage(AigcModel model, ImageGenerationRequest request) {
        int count = request.n() == null || request.n() <= 0 ? DEFAULT_IMAGE_COUNT : request.n();
        // 拼接 :predict 端点，未配置 baseUrl 时使用 Google 官方 v1beta 地址
        String baseUrl = StrUtil.blankToDefault(
                model.getBaseUrl(), "https://generativelanguage.googleapis.com/v1beta");
        String endpoint = StrUtil.removeSuffix(baseUrl, "/")
                + "/models/"
                + model.getModel()
                + ":predict";

        // 组装请求体：instances 携带提示词，parameters.sampleCount 指定生成张数
        Map<String, Object> body = Map.of(
                "instances", List.of(Map.of("prompt", request.prompt())),
                "parameters", Map.of("sampleCount", count)
        );

        // 直连 Gemini 原生协议（非 langchain4j），API Key 通过 x-goog-api-key 请求头传递
        JsonNode response = RestClient.create()
                .post()
                .uri(endpoint)
                .header("x-goog-api-key", defaultApiKey(model))
                .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(JsonNode.class);
        // 校验首张图的 base64 数据存在，缺失则视为生成失败
        if (response == null || !response.has("predictions")
                || response.path("predictions").isEmpty()
                || response.path("predictions").get(0).path("bytesBase64Encoded").asText("").isBlank()) {
            throw new BizException(CoreErrorCode.CHAT_RUNTIME_ERROR);
        }
        // Gemini 仅返回 base64 图片数据，无外部 url
        JsonNode prediction = response.path("predictions").get(0);
        String base64 = prediction.path("bytesBase64Encoded").asText();
        String revisedPrompt = prediction.path("prompt").asText(null);
        return new ImageGenerationResult(null, base64, revisedPrompt);
    }

    /**
     * 图片文字识别（OCR）入口：基于多模态聊天模型实现，
     * 将图片与提示词一起发送给模型，由模型识别并返回文本，
     * 统一记录监控指标（场景 IMAGE_OCR / 类型 OCR）。
     *
     * @param modelId      模型配置 id（需选用支持视觉的多模态模型）
     * @param prompt       识别提示词，为空时使用默认提示词
     * @param imageDataUrl 图片数据（data URL / base64 内联格式）
     * @return 模型识别出的文字内容
     */
    @Override
    public String recognizeText(String modelId, String prompt, String imageDataUrl) {
        // 加载模型配置（识别能力完全依赖所选模型的多模态理解，无独立 OCR 引擎）
        AigcModel model = loadModel(modelId);
        log.info("识别图片文字，modelId={}, model={}", modelId, model.getModel());
        long startTime = System.currentTimeMillis();
        try {
            String text = doRecognizeText(model, prompt, imageDataUrl);
            // 成功则上报监控记录
            modelCallRecorder.record(ModelCallRecord.builder()
                    .metadata(metadataOf(model, ModelCallScene.IMAGE_OCR))
                    .callType(ModelCallType.OCR)
                    .success(true)
                    .itemCount(1)
                    .duration(System.currentTimeMillis() - startTime)
                    .build());
            return text;
        } catch (RuntimeException ex) {
            // 失败同样上报监控（携带异常信息），随后原样抛出由上层处理
            modelCallRecorder.record(ModelCallRecord.builder()
                    .metadata(metadataOf(model, ModelCallScene.IMAGE_OCR))
                    .callType(ModelCallType.OCR)
                    .success(false)
                    .duration(System.currentTimeMillis() - startTime)
                    .errorMessage(ex.getMessage())
                    .build());
            throw ex;
        }
    }

    /**
     * OCR 实际执行：构建多模态聊天模型，发送「文本提示 + 内联图片」的用户消息，
     * 取模型回复文本作为识别结果。
     */
    private String doRecognizeText(AigcModel model, String prompt, String imageDataUrl) {
        // 构建多模态聊天模型（走 OpenAI 兼容协议，支持自定义 baseUrl 覆盖端点）
        OpenAiChatModel.OpenAiChatModelBuilder builder = OpenAiChatModel.builder()
                .modelName(model.getModel())
                .apiKey(defaultApiKey(model))
                .timeout(timeout(model));
        applyBaseUrl(model, builder::baseUrl);
        OpenAiChatModel chatModel = builder.build();

        // 提示词为空时回退默认 OCR 提示；图片以 data URL（base64 内联）形式传入
        String userPrompt = StrUtil.blankToDefault(prompt, DEFAULT_OCR_PROMPT);
        UserMessage userMessage = UserMessage.from(
                TextContent.from(userPrompt),
                ImageContent.from(imageDataUrl)
        );
        // 发起单轮多模态对话，取 AI 回复文本作为识别结果
        ChatResponse response = chatModel.chat(ChatRequest.builder()
                .messages(List.of(userMessage))
                .build());
        if (response == null || response.aiMessage() == null) {
            throw new BizException(CoreErrorCode.CHAT_RUNTIME_ERROR);
        }
        return response.aiMessage().text();
    }

    /**
     * 加载模型配置，不存在则抛出异常。
     */
    private AigcModel loadModel(String modelId) {
        if (StrUtil.isBlank(modelId)) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND);
        }
        AigcModel model = aigcModelService.getById(modelId);
        if (model == null) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND);
        }
        return model;
    }

    /** 获取 API Key：未配置时返回占位值 EMPTY_API_KEY，避免模型构建器的非空校验失败。 */
    private String defaultApiKey(AigcModel model) {
        return StrUtil.isBlank(model.getApiKey()) ? EMPTY_API_KEY : model.getApiKey();
    }

    /** 解析请求超时：模型未配置或配置非正数时回退为默认 3 分钟。 */
    private Duration timeout(AigcModel model) {
        int minutes = model.getTimeout() == null || model.getTimeout() <= 0 ? 3 : model.getTimeout();
        return Duration.ofMinutes(minutes);
    }

    /**
     * 条件式应用自定义 baseUrl：模型配置了 baseUrl 时才调用 consumer 覆盖端点，
     * 未配置则保留 SDK 默认地址，供各模型构建器以方法引用方式复用。
     */
    private void applyBaseUrl(AigcModel model, java.util.function.Consumer<String> consumer) {
        if (StrUtil.isNotBlank(model.getBaseUrl())) {
            consumer.accept(model.getBaseUrl());
        }
    }
}
