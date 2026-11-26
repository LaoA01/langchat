package cn.langchat.core.chat.service.impl;

import cn.langchat.aigc.biz.entity.AigcDocs;
import cn.langchat.aigc.biz.entity.AigcKnowledge;
import cn.langchat.aigc.biz.entity.AigcModel;
import cn.langchat.aigc.biz.entity.AigcSegment;
import cn.langchat.aigc.biz.entity.AigcVectorStore;
import cn.langchat.aigc.biz.service.AigcDocsService;
import cn.langchat.aigc.biz.service.AigcKnowledgeService;
import cn.langchat.aigc.biz.service.AigcModelService;
import cn.langchat.aigc.biz.service.AigcSegmentService;
import cn.langchat.aigc.biz.service.AigcVectorStoreService;
import cn.langchat.aigc.biz.support.ModelConfigSupport;
import cn.langchat.common.exception.BizException;
import cn.langchat.core.chat.enums.DocumentIndexStatusEnum;
import cn.langchat.core.chat.model.parse.ParsedDocumentContent;
import cn.langchat.core.chat.service.KnowledgeDocumentIndexProcessor;
import cn.langchat.core.runtime.factory.LangChain4jModelFactory;
import cn.langchat.core.runtime.factory.VectorStoreFactory;
import cn.langchat.core.runtime.rag.DocumentChunkingService;
import cn.langchat.core.runtime.rag.DocumentContentParseService;
import cn.langchat.core.runtime.rag.SegmentMetadataFactory;
import cn.langchat.core.support.CoreErrorCode;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * 单文档知识索引处理器实现。
 *
 * @author LangChat Team
 * @since 2026/3/25
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class KnowledgeDocumentIndexProcessorImpl implements KnowledgeDocumentIndexProcessor {

    /** 落库错误信息截断上限，防止 DB 字段溢出。 */
    private static final int MAX_ERROR_LENGTH = 1000;

    private final AigcKnowledgeService aigcKnowledgeService;
    private final AigcDocsService aigcDocsService;
    private final AigcSegmentService aigcSegmentService;
    private final AigcVectorStoreService aigcVectorStoreService;
    private final AigcModelService aigcModelService;
    private final DocumentChunkingService documentChunkingService;
    private final DocumentContentParseService documentContentParseService;
    private final SegmentMetadataFactory segmentMetadataFactory;
    private final LangChain4jModelFactory langChain4jModelFactory;
    private final VectorStoreFactory vectorStoreFactory;

    /**
     * 单文档索引的异步执行体（由 KnowledgeIndexAsyncServiceImpl 派发到专用线程池调用）。
     *
     * <p>完整链路：校验资源 → 标记运行中 → 清理旧分段 → 解析切片 → 分段落库 →
     * 批量向量化写入向量库 → 更新终态。任何环节失败都会把文档标记为 FAILED、
     * 记录耗时与错误信息，随后重新抛出异常交由异步框架记录日志。
     * chunkSize/overlapSize 传 null 时由切分服务按内置默认策略执行。</p>
     */
    @Override
    public void processDocument(String knowledgeId, String docsId, Integer chunkSize, Integer overlapSize) {
        long startTime = System.currentTimeMillis();
        // 1. 加载并校验文档：必须存在且归属当前知识库，防止跨库误索引
        AigcDocs docs = requireDocs(docsId, knowledgeId);
        log.info("开始异步处理文档向量化任务，knowledgeId={}, docsId={}", knowledgeId, docsId);
        // 2. 立即标记运行中并记录开始时间，前端 queryIndexStatus 轮询可见进度切换
        updateDocsStatus(docs, DocumentIndexStatusEnum.RUNNING, null, startTime, null);
        try {
            // 3. 加载知识库与依赖资源（向量库配置、向量模型配置），配置缺失即刻报错
            AigcKnowledge knowledge = requireKnowledge(knowledgeId);
            AigcVectorStore vectorStore = loadVectorStore(knowledge);
            AigcModel embeddingModelConfig = loadEmbeddingModel(knowledge);
            // 3.1 维度强校验：模型与向量库维度不一致会导致写入失败或索引污染，此环节尽早拦截
            validateEmbeddingDimension(embeddingModelConfig, vectorStore);
            // 4. 获取向量库客户端（按类型路由）与向量模型（走工厂缓存），由配置转化为可调用实例
            EmbeddingStore<TextSegment> embeddingStore = vectorStoreFactory.getStore(vectorStore);
            EmbeddingModel embeddingModel = langChain4jModelFactory.getEmbeddingModel(embeddingModelConfig);

            // 5. 删除该文档旧分段，保证重复索引/重试从干净状态重建（业务侧切片记录）
            aigcSegmentService.lambdaUpdate().eq(AigcSegment::getDocsId, docsId).remove();
            // 6. 解析文档 → 按切片配置切分 → 构建 AigcSegment 实体
            List<AigcSegment> segments = createSegments(docs, chunkSize, overlapSize);
            // 7. 分段批量入库：作为检索结果展示与溯源的业务依据
            aigcSegmentService.saveBatch(segments);
            // 8. 批量向量化并写入向量库（内部按分段 id 先删后写保证幂等）
            embedSegments(knowledge, docs, segments, embeddingModel, embeddingStore);

            long endTime = System.currentTimeMillis();
            // 9. 标记完成：记录总耗时，清空错误信息
            updateDocsStatus(docs, DocumentIndexStatusEnum.COMPLETED, null, startTime, endTime);
            log.info(
                    "文档向量化完成，knowledgeId={}, docsId={}, segmentCount={}, costMs={}",
                    knowledgeId,
                    docsId,
                    segments.size(),
                    endTime - startTime
            );
        } catch (Exception ex) {
            long endTime = System.currentTimeMillis();
            // 失败兜底：规整（截断/兜底文案）错误信息后落库并记录耗时
            String errorMessage = resolveErrorMessage(ex);
            // 标记失败状态，前端轮询可见；随后重抛异常交由异步框架记录日志
            updateDocsStatus(docs, DocumentIndexStatusEnum.FAILED, errorMessage, startTime, endTime);
            log.error(
                    "文档向量化失败，knowledgeId={}, docsId={}, costMs={}, error={}",
                    knowledgeId,
                    docsId,
                    endTime - startTime,
                    errorMessage,
                    ex
            );
            throw ex;
        }
    }

    /** 加载并校验文档：必须存在且归属指定知识库，缺失或跨库即刻报错，防止误索引。 */
    private AigcDocs requireDocs(String docsId, String knowledgeId) {
        AigcDocs docs = aigcDocsService.getById(docsId);
        // 文档不存在，或属于其他知识库——两种情况均视为无效请求
        if (docs == null || !Objects.equals(knowledgeId, docs.getKnowledgeId())) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND.code(), "文档不存在或不属于当前知识库");
        }
        return docs;
    }

    /** 加载并校验知识库存在，缺失时直接报错。 */
    private AigcKnowledge requireKnowledge(String knowledgeId) {
        AigcKnowledge knowledge = aigcKnowledgeService.getById(knowledgeId);
        if (knowledge == null) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND);
        }
        return knowledge;
    }

    /** 加载向量库配置：向量必须有落库目标，知识库未绑定或配置已删除即刻报错。 */
    private AigcVectorStore loadVectorStore(AigcKnowledge knowledge) {
        if (knowledge.getVectorStoreId() == null || knowledge.getVectorStoreId().isBlank()) {
            throw new BizException(CoreErrorCode.VECTOR_STORE_NOT_FOUND);
        }
        AigcVectorStore vectorStore = aigcVectorStoreService.getById(knowledge.getVectorStoreId());
        if (vectorStore == null) {
            throw new BizException(CoreErrorCode.VECTOR_STORE_NOT_FOUND);
        }
        return vectorStore;
    }

    /** 加载向量模型配置：向量化依赖嵌入模型，知识库未配置或模型已删除即刻报错。 */
    private AigcModel loadEmbeddingModel(AigcKnowledge knowledge) {
        String modelId = knowledge.getVectorModelId();
        if (modelId == null || modelId.isBlank()) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND.code(), "知识库缺少向量模型配置");
        }
        AigcModel model = aigcModelService.getById(modelId);
        if (model == null) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND);
        }
        return model;
    }

    /**
     * 校验向量模型与向量库维度一致。
     *
     * <p>维度不匹配时写入向量库会直接报错或污染既有索引，此环节在真正调模型前
     * 强校验，把配置问题尽早暴露；同时要求模型必须显式配置 dimension。</p>
     */
    private void validateEmbeddingDimension(AigcModel model, AigcVectorStore vectorStore) {
        Integer modelDimension = ModelConfigSupport.resolveDimension(model);
        if (modelDimension == null) {
            throw new BizException(
                    CoreErrorCode.INVALID_MODEL_CONFIG.code(),
                    "向量模型 configJson.dimension 必须配置为大于 0 的整数"
            );
        }
        Integer storeDimension = vectorStore.getDimension();
        if (!Objects.equals(modelDimension, storeDimension)) {
            throw new BizException(
                    CoreErrorCode.EMBEDDING_DIMENSION_MISMATCH.code(),
                    "向量模型维度 " + modelDimension + " 与向量库维度 " + storeDimension + " 不一致"
            );
        }
    }

    /**
     * 解析文档并按切片配置生成分段实体。
     *
     * <p>流程：解析器按文件类型解析出 sections → 逐段切分为 chunks →
     * 构建 AigcSegment 实体。indexHash 由「docsId + 位置 + 内容」摘要，
     * 后续可用于内容变更检测；解析结果为空则报错，避免向向量库写入空数据。</p>
     */
    private List<AigcSegment> createSegments(AigcDocs docs, Integer chunkSize, Integer overlapSize) {
        // 解析文档正文为分段，由解析服务按文件类型路由对应解析器
        ParsedDocumentContent parsedDocument = documentContentParseService.parse(docs);
        List<String> chunks = new ArrayList<>();
        // 逐段切片；chunkSize/overlapSize 为 null 时由切分服务按内置默认策略
        for (String section : parsedDocument.getSections()) {
            chunks.addAll(documentChunkingService.chunk(section, chunkSize, overlapSize));
        }
        // 解析/切分后无任何内容，视为无效文档，显式报错而非静默入库
        if (chunks.isEmpty()) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND.code(), "文档解析结果为空，无法生成向量切片");
        }
        List<AigcSegment> segments = new ArrayList<>();
        // 逐 chunk 构建实体：UUID 作为分段主键，position 记录文档内顺序，
        // indexHash 用于内容指纹（变更检测），status=1 表示初始有效态
        for (int index = 0; index < chunks.size(); index++) {
            AigcSegment segment = new AigcSegment();
            segment.setId(UUID.randomUUID().toString());
            segment.setDocsId(docs.getId());
            segment.setKnowledgeId(docs.getKnowledgeId());
            segment.setEnabled(Boolean.TRUE);
            segment.setName(docs.getName());
            segment.setPosition(index);
            segment.setContent(chunks.get(index));
            segment.setIndexHash(Integer.toHexString(Objects.hash(docs.getId(), index, chunks.get(index))));
            segment.setStatus(1);
            segments.add(segment);
        }
        return segments;
    }

    /**
     * 批量向量化并写入向量库，按分段 id 先删后写保证重复索引幂等。
     *
     * <p>TextSegment 携带由 SegmentMetadataFactory 生成的元数据（知识库/文档/分段信息），
     * 检索命中时可溯源；embedAll 一次调用批量生成向量，减少模型 API 往返；
     * 随后按分段 id 先 removeAll 再 addAll，避免重试导致向量库出现重复向量。</p>
     */
    private void embedSegments(
            AigcKnowledge knowledge,
            AigcDocs docs,
            List<AigcSegment> segments,
            EmbeddingModel embeddingModel,
            EmbeddingStore<TextSegment> embeddingStore
    ) {
        // 构建带元数据的文本段，元数据在检索命中时用于溯源（对应知识库检索结果的展示字段）
        List<TextSegment> textSegments = segments.stream()
                .map(segment -> TextSegment.from(segment.getContent(), segmentMetadataFactory.build(knowledge, docs, segment)))
                .toList();
        // 一次性批量向量化，减少调模型次数
        Response<List<Embedding>> response = embeddingModel.embedAll(textSegments);
        List<Embedding> embeddings = response.content();
        // 以分段主键作为向量库中的 id，实现可寻址的先删后写
        List<String> ids = segments.stream().map(AigcSegment::getId).toList();
        // 先删旧向量再加入新向量：重试/重复索引时同 id 向量被替换，不会产生重复
        embeddingStore.removeAll(ids);
        embeddingStore.addAll(ids, embeddings, textSegments);
    }

    /**
     * 统一更新文档索引状态（内存对象 + DB 双写），同时记录开始/结束时间与错误信息。
     * 被 RUNNING / COMPLETED / FAILED 三处复用，是前端 queryIndexStatus 轮询的状态来源。
     */
    private void updateDocsStatus(
            AigcDocs docs,
            DocumentIndexStatusEnum status,
            String errorMessage,
            Long startTime,
            Long endTime
    ) {
        long now = System.currentTimeMillis();
        // 原地更新内存对象，保证同一流程内后续读取拿到最新值
        docs.setIndexingStatus(status.dbStatus());
        docs.setEmbedStatus(status.code());
        docs.setEmbedError(errorMessage);
        docs.setEmbedStartTime(startTime);
        docs.setEmbedEndTime(endTime);
        docs.setUpdateTime(now);
        // 持久化到 DB，前端轮询可见状态变化
        aigcDocsService.lambdaUpdate()
                .eq(AigcDocs::getId, docs.getId())
                .set(AigcDocs::getIndexingStatus, status.dbStatus())
                .set(AigcDocs::getEmbedStatus, status.code())
                .set(AigcDocs::getEmbedError, errorMessage)
                .set(AigcDocs::getEmbedStartTime, startTime)
                .set(AigcDocs::getEmbedEndTime, endTime)
                .set(AigcDocs::getUpdateTime, now)
                .update();
    }

    /** 规整错误信息：空值兜底为通用文案、超长按 MAX_ERROR_LENGTH 截断，避免 DB 字段溢出。 */
    private String resolveErrorMessage(Exception ex) {
        String message = ex.getMessage();
        if (message == null || message.isBlank()) {
            return "文档向量化失败";
        }
        if (message.length() <= MAX_ERROR_LENGTH) {
            return message;
        }
        return message.substring(0, MAX_ERROR_LENGTH);
    }
}
