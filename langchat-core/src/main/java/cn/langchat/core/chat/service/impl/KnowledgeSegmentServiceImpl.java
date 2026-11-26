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
import cn.langchat.common.exception.BizException;
import cn.langchat.core.chat.model.request.SegmentContentUpdateRequest;
import cn.langchat.core.chat.model.request.SegmentDeleteRequest;
import cn.langchat.core.chat.model.request.SegmentEnabledRequest;
import cn.langchat.core.chat.service.KnowledgeSegmentService;
import cn.langchat.core.runtime.factory.LangChain4jModelFactory;
import cn.langchat.core.runtime.factory.VectorStoreFactory;
import cn.langchat.core.runtime.rag.SegmentMetadataFactory;
import cn.langchat.core.support.CoreErrorCode;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * 知识库分段运行时服务实现。
 *
 * <p>分段检索直接命中向量库，因此分段的启停、编辑与删除都必须同步维护向量数据，
 * 否则数据库与向量库会出现不一致。</p>
 *
 * @author LangChat Team
 * @since 2026/9/27
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class KnowledgeSegmentServiceImpl implements KnowledgeSegmentService {

    /** 知识库服务：校验知识库存在性。 */
    private final AigcKnowledgeService aigcKnowledgeService;
    /** 文档服务：校验文档归属关系。 */
    private final AigcDocsService aigcDocsService;
    /** 分段服务：分段增删改查。 */
    private final AigcSegmentService aigcSegmentService;
    /** 模型服务：解析向量模型配置。 */
    private final AigcModelService aigcModelService;
    /** 向量库配置服务：按知识库查找向量库实体。 */
    private final AigcVectorStoreService aigcVectorStoreService;
    /** 模型工厂：构建向量模型实例（带监控监听）。 */
    private final LangChain4jModelFactory langChain4jModelFactory;
    /** 向量库工厂：按配置构建 EmbeddingStore。 */
    private final VectorStoreFactory vectorStoreFactory;
    /** 分段元数据工厂：为 TextSegment 构造检索元数据（来源、位置等）。 */
    private final SegmentMetadataFactory segmentMetadataFactory;

    /**
     * 查询指定文档的分段列表：校验文档归属后按 position 升序返回。
     *
     * @param knowledgeId 知识库 ID
     * @param docsId      文档 ID
     * @return 分段列表（按文档内顺序排列）
     */
    @Override
    public List<AigcSegment> listSegments(String knowledgeId, String docsId) {
        // 校验文档存在且属于该知识库
        AigcDocs docs = requireDocs(knowledgeId, docsId);
        log.info("查询文档分段列表，knowledgeId={}, docsId={}", knowledgeId, docs.getId());
        return aigcSegmentService.lambdaQuery()
                .eq(AigcSegment::getDocsId, docs.getId())
                .orderByAsc(AigcSegment::getPosition)
                .list();
    }

    /**
     * 更新分段文本内容：仅修改数据库记录，不自动重新向量化
     * （需调用方明确调用 reindexSegment 触发重新入库，避免高频操作压垮向量模型）。
     */
    @Override
    public void updateSegmentContent(String knowledgeId, String segmentId, SegmentContentUpdateRequest request) {
        // 校验分段存在且属于该知识库
        AigcSegment segment = requireSegment(knowledgeId, segmentId);
        String content = request == null ? null : request.getContent();
        // 内容为空不合法
        if (content == null || content.isBlank()) {
            throw new BizException(CoreErrorCode.INVALID_SEGMENT_REQUEST.code(), "分段内容不能为空");
        }
        aigcSegmentService.lambdaUpdate()
                .eq(AigcSegment::getId, segment.getId())
                .set(AigcSegment::getContent, content)
                .update();
        log.info("更新分段内容，docsId={}, segmentId={}, contentLength={}", segment.getDocsId(), segmentId, content.length());
    }

    /**
     * 启用/停用分段，同步维护向量数据：启用需先写向量（embedSegments），
     * 停用则从向量库移除该分段向量；请求未指定 enabled 时默认启用。
     */
    @Override
    public void updateSegmentEnabled(String knowledgeId, String segmentId, SegmentEnabledRequest request) {
        AigcSegment segment = requireSegment(knowledgeId, segmentId);
        // 缺省置为启用
        boolean enabled = request == null || request.getEnabled() == null ? Boolean.TRUE : request.getEnabled();
        // 加载向量化上下文（向量库 + 向量模型），供写/删向量
        SegmentEmbeddingContext context = loadEmbeddingContext(knowledgeId, segment.getDocsId());
        aigcSegmentService.lambdaUpdate()
                .eq(AigcSegment::getId, segment.getId())
                .set(AigcSegment::getEnabled, enabled)
                .update();
        if (enabled) {
            // 启用：写回向量（含先删后加，保证元数据最新）
            embedSegments(context, List.of(segment));
            log.info("启用分段并写入向量，knowledgeId={}, segmentId={}", knowledgeId, segmentId);
            return;
        }
        // 停用：直接从向量库移除，向量库不再命中该分段
        context.embeddingStore().removeAll(List.of(segmentId));
        log.info("停用分段并移除向量，knowledgeId={}, segmentId={}", knowledgeId, segmentId);
    }

    /**
     * 重新向量化单个分段：要求分段处于启用状态（停用分段需先启用），
     * 否则向量写入无意义；通过 embedSegments 先删后加完成更新。
     */
    @Override
    public void reindexSegment(String knowledgeId, String segmentId) {
        // 校验分段存在且属于该知识库
        AigcSegment segment = requireSegment(knowledgeId, segmentId);
        // 停用分段不能入库，否则检索会命中不可见内容
        if (!Objects.equals(Boolean.TRUE, segment.getEnabled())) {
            throw new BizException(CoreErrorCode.INVALID_SEGMENT_REQUEST.code(), "分段已停用，请先启用后再重新向量化");
        }
        SegmentEmbeddingContext context = loadEmbeddingContext(knowledgeId, segment.getDocsId());
        embedSegments(context, List.of(segment));
        log.info("完成单个分段重新向量化，knowledgeId={}, docsId={}, segmentId={}", knowledgeId, segment.getDocsId(), segmentId);
    }

    /** 删除单个分段：复用批量删除逻辑，内部先清向量再删数据库记录。 */
    @Override
    public void deleteSegment(String knowledgeId, String segmentId) {
        // 校验后复用批量删除，保证向量同步清理
        requireSegment(knowledgeId, segmentId);
        SegmentDeleteRequest request = new SegmentDeleteRequest();
        request.setSegmentIds(List.of(segmentId));
        deleteSegments(knowledgeId, request);
    }

    /**
     * 批量删除分段：去空去重后查询命中分段，先清理向量再删除数据库记录——
     * 向量库不可用时保留数据库记录，用户可重试，避免出现数据库无数据但向量仍可检索的不一致。
     */
    @Override
    public void deleteSegments(String knowledgeId, SegmentDeleteRequest request) {
        List<String> segmentIds = request == null || request.getSegmentIds() == null
                ? List.of()
                : request.getSegmentIds();
        // 批量参数去空、去重（保序）
        Set<String> uniqueIds = new LinkedHashSet<>(segmentIds);
        uniqueIds.removeIf(id -> id == null || id.isBlank());
        if (uniqueIds.isEmpty()) {
            throw new BizException(CoreErrorCode.INVALID_SEGMENT_REQUEST.code(), "请先选择要删除的分段");
        }
        // 查询当前知识库下真实存在的分段，防越权删除其他知识库数据
        List<AigcSegment> segments = aigcSegmentService.lambdaQuery()
                .eq(AigcSegment::getKnowledgeId, knowledgeId)
                .in(AigcSegment::getId, uniqueIds)
                .list();
        if (segments.isEmpty()) {
            throw new BizException(CoreErrorCode.SEGMENT_NOT_FOUND.code(), "分段不存在或不属于当前知识库");
        }
        List<String> removedIds = segments.stream().map(AigcSegment::getId).toList();
        // 先清理向量再删除记录：向量库不可用时保留数据库记录，用户可以重试，不会出现库里有数据但检索不到
        SegmentEmbeddingContext context = loadEmbeddingContext(knowledgeId, segments.get(0).getDocsId());
        context.embeddingStore().removeAll(removedIds);
        aigcSegmentService.removeByIds(removedIds);
        log.info("删除分段并清理向量，knowledgeId={}, segmentCount={}", knowledgeId, removedIds.size());
    }

    /** 校验并加载文档：先确认知识库存在，再确认文档存在且属于该知识库。 */
    private AigcDocs requireDocs(String knowledgeId, String docsId) {
        AigcKnowledge knowledge = aigcKnowledgeService.getById(knowledgeId);
        if (knowledge == null) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND);
        }
        AigcDocs docs = aigcDocsService.getById(docsId);
        if (docs == null || !Objects.equals(knowledgeId, docs.getKnowledgeId())) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND.code(), "文档不存在或不属于当前知识库");
        }
        return docs;
    }

    /** 校验并加载分段：分段不存在或不属于当前知识库均视为未找到，防越权操作。 */
    private AigcSegment requireSegment(String knowledgeId, String segmentId) {
        AigcSegment segment = aigcSegmentService.getById(segmentId);
        if (segment == null || !Objects.equals(knowledgeId, segment.getKnowledgeId())) {
            throw new BizException(CoreErrorCode.SEGMENT_NOT_FOUND.code(), "分段不存在或不属于当前知识库");
        }
        return segment;
    }

    /**
     * 加载分段向量化上下文：依次校验知识库、文档、向量库配置与向量模型配置，
     * 并确认绑定模型类型为 EMBEDDING；全部通过后构建 EmbeddingStore 与 EmbeddingModel 实例。
     *
     * @return 向量化上下文（知识库、文档、向量库实例、向量模型实例）
     */
    private SegmentEmbeddingContext loadEmbeddingContext(String knowledgeId, String docsId) {
        // 知识库必须存在
        AigcKnowledge knowledge = aigcKnowledgeService.getById(knowledgeId);
        if (knowledge == null) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND);
        }
        // 文档必须存在
        AigcDocs docs = aigcDocsService.getById(docsId);
        if (docs == null) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND.code(), "文档不存在或已被删除");
        }
        // 知识库必须已绑定向量库且向量库记录存在
        if (knowledge.getVectorStoreId() == null || knowledge.getVectorStoreId().isBlank()) {
            throw new BizException(CoreErrorCode.VECTOR_STORE_NOT_FOUND);
        }
        AigcVectorStore vectorStore = aigcVectorStoreService.getById(knowledge.getVectorStoreId());
        if (vectorStore == null) {
            throw new BizException(CoreErrorCode.VECTOR_STORE_NOT_FOUND);
        }
        // 知识库必须绑定类型为 EMBEDDING 的向量模型，且模型记录存在
        if (knowledge.getVectorModelId() == null || knowledge.getVectorModelId().isBlank()) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND.code(), "知识库缺少向量模型配置");
        }
        AigcModel embeddingModelConfig = aigcModelService.getById(knowledge.getVectorModelId());
        if (embeddingModelConfig == null) {
            throw new BizException(CoreErrorCode.MODEL_NOT_FOUND);
        }
        // 类型校验：聊天/推理模型不能当向量模型用，给出可操作的提示
        if (!Objects.equals("EMBEDDING", embeddingModelConfig.getType())) {
            throw new BizException(
                    CoreErrorCode.MODEL_NOT_FOUND.code(),
                    "知识库绑定的「" + embeddingModelConfig.getName() + "」不是向量模型，请先在模型管理中创建向量模型并重新绑定"
            );
        }
        // 构建向量库与向量模型实例，组装上下文
        return new SegmentEmbeddingContext(
                knowledge,
                docs,
                vectorStoreFactory.getStore(vectorStore),
                langChain4jModelFactory.getEmbeddingModel(embeddingModelConfig)
        );
    }

    /**
     * 分段写向量：内容 + 元数据 → 生成 TextSegment → 逐个向量量化 →
     * 先按 id 移除旧向量再批量写入新向量（先删后加，保证修改后重写入库的幂等性）。
     */
    private void embedSegments(SegmentEmbeddingContext context, List<AigcSegment> segments) {
        if (segments.isEmpty()) {
            return;
        }
        // 构造 TextSegment：正文 + 来源元数据（知识库/文档/分段信息）
        List<TextSegment> textSegments = new ArrayList<>(segments.size());
        for (AigcSegment segment : segments) {
            textSegments.add(TextSegment.from(segment.getContent(), segmentMetadataFactory.build(context.knowledge(), context.docs(), segment)));
        }
        // 逐段计算向量
        List<Embedding> embeddings = new ArrayList<>(segments.size());
        for (TextSegment textSegment : textSegments) {
            embeddings.add(context.embeddingModel().embed(textSegment).content());
        }
        // 先删后加：移除同 id 旧向量再批量写入，保证多次调用结果一致
        List<String> ids = segments.stream().map(AigcSegment::getId).toList();
        context.embeddingStore().removeAll(ids);
        context.embeddingStore().addAll(ids, embeddings, textSegments);
    }

    /**
     * 分段向量化上下文。
     *
     * @param knowledge 知识库
     * @param docs 文档
     * @param embeddingStore 向量库实例
     * @param embeddingModel 向量模型实例
     * @author LangChat Team
     * @since 2026/9/27
     */
    private record SegmentEmbeddingContext(
            AigcKnowledge knowledge,
            AigcDocs docs,
            EmbeddingStore<TextSegment> embeddingStore,
            EmbeddingModel embeddingModel
    ) {
    }
}
