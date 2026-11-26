package cn.langchat.core.chat.service.impl;

import cn.langchat.aigc.biz.entity.AigcDocs;
import cn.langchat.aigc.biz.entity.AigcKnowledge;
import cn.langchat.aigc.biz.service.AigcDocsService;
import cn.langchat.aigc.biz.service.AigcKnowledgeService;
import cn.langchat.aigc.biz.service.AigcSegmentService;
import cn.langchat.common.exception.BizException;
import cn.langchat.core.chat.enums.DocumentIndexStatusEnum;
import cn.langchat.core.chat.model.parse.ParsedDocumentContent;
import cn.langchat.core.chat.model.request.KnowledgeIndexRequest;
import cn.langchat.core.chat.model.request.KnowledgeParsePreviewRequest;
import cn.langchat.core.chat.model.response.KnowledgeDocumentPreview;
import cn.langchat.core.chat.model.response.KnowledgeIndexResult;
import cn.langchat.core.chat.model.response.KnowledgeDocumentIndexStatus;
import cn.langchat.core.chat.model.response.KnowledgeIndexStatusResult;
import cn.langchat.core.chat.model.response.KnowledgeParsePreviewResult;
import cn.langchat.core.chat.service.KnowledgeIndexAsyncService;
import cn.langchat.core.chat.service.KnowledgeIndexRuntimeService;
import cn.langchat.core.runtime.rag.DocumentContentParseService;
import cn.langchat.core.runtime.rag.DocumentChunkingService;
import cn.langchat.core.support.CoreErrorCode;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * 知识库索引运行时服务实现。
 *
 * @author LangChat Team
 * @since 2026/3/25
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class KnowledgeIndexRuntimeServiceImpl implements KnowledgeIndexRuntimeService {

    private static final int DEFAULT_PREVIEW_SECTION_LIMIT = 5;
    private static final int DEFAULT_PREVIEW_CHUNK_LIMIT = 10;
    private static final String CONFIG_CHUNK_SIZE = "chunkSize";
    private static final String CONFIG_CHUNK_SIZE_ALT = "chunk_size";
    private static final String CONFIG_OVERLAP_SIZE = "overlapSize";
    private static final String CONFIG_OVERLAP_SIZE_ALT = "overlap_size";
    private static final TypeReference<Map<String, Object>> MAP_TYPE = new TypeReference<>() {
    };
    private static final SegmentSummary EMPTY_SEGMENT_SUMMARY = new SegmentSummary(0, 0L);

    private final AigcKnowledgeService aigcKnowledgeService;
    private final AigcDocsService aigcDocsService;
    private final AigcSegmentService aigcSegmentService;
    private final DocumentChunkingService documentChunkingService;
    private final DocumentContentParseService documentContentParseService;
    private final KnowledgeIndexAsyncService knowledgeIndexAsyncService;
    private final ObjectMapper objectMapper;

    /**
     * 提交知识库索引任务：将待索引文档逐个交给异步线程池向量化，
     * 本方法只负责「圈定文档 + 重置状态 + 派发任务」并立即返回，
     * 实际的解析、切片、向量化在后台完成，前端通过 queryIndexStatus 轮询进度。
     */
    @Override
    public KnowledgeIndexResult indexKnowledge(String knowledgeId, KnowledgeIndexRequest request) {
        // 1. 校验知识库存在，不存在直接报错（避免对孤儿文档派发无效任务）
        AigcKnowledge knowledge = aigcKnowledgeService.getById(knowledgeId);
        if (knowledge == null) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND);
        }
        // 2. 加载待索引文档（按请求 docsIds 圈定，排除已停用文档）
        List<AigcDocs> docsList = loadDocs(knowledgeId, request);
        // 3. 逐个文档派发：解析切片参数 → 重置为排队状态 → 提交异步索引任务
        for (AigcDocs docs : docsList) {
            // 3.1 解析切片配置：优先请求参数，其次文档 ingestionConfig，最终回退处理器默认
            DocumentSplitConfig splitConfig = resolveSplitConfig(docs, request.getChunkSize(), request.getOverlapSize());
            // 3.2 将文档状态重置为 PENDING，清空历史失败/完成痕迹，避免状态查询误判
            markDocsQueued(docs);
            // 3.3 提交到异步线程池（本调用立即返回，不阻塞），由处理器完成解析→切片→向量化→入库
            knowledgeIndexAsyncService.submitDocumentIndex(
                    knowledgeId,
                    docs.getId(),
                    splitConfig.chunkSize(),
                    splitConfig.overlapSize()
            );
            log.info(
                    "提交知识库文档索引任务，knowledgeId={}, docsId={}, chunkSize={}, overlapSize={}",
                    knowledgeId,
                    docs.getId(),
                    splitConfig.chunkSize(),
                    splitConfig.overlapSize()
            );
        }

        // 4. 汇总返回：因任务异步执行，此处 segmentCount/embeddedCount 恒为 0，
        //    真实进度需调用 queryIndexStatus 获取
        KnowledgeIndexResult result = new KnowledgeIndexResult();
        result.setKnowledgeId(knowledgeId);
        result.setDocsCount(docsList.size());
        result.setSubmittedCount(docsList.size());
        result.setAsync(Boolean.TRUE);
        result.setTaskStatus(DocumentIndexStatusEnum.PENDING.code());
        result.setSegmentCount(0);
        result.setEmbeddedCount(0);
        log.info("知识库索引任务已异步提交，knowledgeId={}, docsCount={}", knowledgeId, docsList.size());
        return result;
    }

    @Override
    public KnowledgeParsePreviewResult previewKnowledge(String knowledgeId, KnowledgeParsePreviewRequest request) {
        AigcKnowledge knowledge = aigcKnowledgeService.getById(knowledgeId);
        if (knowledge == null) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND);
        }
        List<AigcDocs> docsList = loadDocs(knowledgeId, buildIndexRequest(request));
        List<KnowledgeDocumentPreview> previews = docsList.stream()
                .map(docs -> buildPreview(docs, request))
                .toList();
        KnowledgeParsePreviewResult result = new KnowledgeParsePreviewResult();
        result.setKnowledgeId(knowledgeId);
        result.setDocsCount(previews.size());
        result.setDocs(previews);
        log.info("知识库解析预览完成，knowledgeId={}, docsCount={}", knowledgeId, previews.size());
        return result;
    }

    @Override
    public KnowledgeIndexStatusResult queryIndexStatus(String knowledgeId, List<String> docsIds) {
        AigcKnowledge knowledge = aigcKnowledgeService.getById(knowledgeId);
        if (knowledge == null) {
            throw new BizException(CoreErrorCode.KNOWLEDGE_NOT_FOUND);
        }
        List<AigcDocs> docsList = aigcDocsService.lambdaQuery()
                .eq(AigcDocs::getKnowledgeId, knowledgeId)
                .in(docsIds != null && !docsIds.isEmpty(), AigcDocs::getId, docsIds)
                .orderByDesc(AigcDocs::getUpdateTime)
                .list();
        long now = System.currentTimeMillis();
        Map<String, SegmentSummary> segmentSummary = loadSegmentSummary(docsList);
        List<KnowledgeDocumentIndexStatus> docStatuses = docsList.stream()
                .map(docs -> buildDocumentIndexStatus(docs, now, segmentSummary))
                .toList();

        int pendingCount = countByStatus(docStatuses, DocumentIndexStatusEnum.PENDING.code());
        int runningCount = countByStatus(docStatuses, DocumentIndexStatusEnum.RUNNING.code());
        int completedCount = countByStatus(docStatuses, DocumentIndexStatusEnum.COMPLETED.code());
        int failedCount = countByStatus(docStatuses, DocumentIndexStatusEnum.FAILED.code());

        KnowledgeIndexStatusResult result = new KnowledgeIndexStatusResult();
        result.setKnowledgeId(knowledgeId);
        result.setDocsCount(docStatuses.size());
        result.setPendingCount(pendingCount);
        result.setRunningCount(runningCount);
        result.setCompletedCount(completedCount);
        result.setFailedCount(failedCount);
        result.setFinished(docStatuses.isEmpty() || (pendingCount == 0 && runningCount == 0));
        result.setProgressPercent(calculateProgressPercent(docStatuses.size(), completedCount, failedCount));
        result.setDocs(docStatuses);
        log.info(
                "知识库索引状态查询完成，knowledgeId={}, docsCount={}, pendingCount={}, runningCount={}, completedCount={}, failedCount={}",
                knowledgeId,
                docStatuses.size(),
                pendingCount,
                runningCount,
                completedCount,
                failedCount
        );
        return result;
    }

    /**
     * 加载知识库下待索引的文档列表。
     *
     * <p>过滤规则：只取启用（enabled=true 或未被显式停用）的文档；
     * 若请求指定了 docsIds 则进一步按 ID 圈定，未指定则返回全部启用文档。
     * 按更新时间倒序排列，最新修改的文档优先索引。列表可能为空，不视为异常。</p>
     */
    private List<AigcDocs> loadDocs(String knowledgeId, KnowledgeIndexRequest request) {
        return aigcDocsService.lambdaQuery()
                .eq(AigcDocs::getKnowledgeId, knowledgeId)
                // 仅取启用文档：enabled=true 或字段为 null（历史数据未显式停用）
                .and(query -> query.eq(AigcDocs::getEnabled, Boolean.TRUE).or().isNull(AigcDocs::getEnabled))
                // 可选圈定：请求带 docsIds 时过滤，条件表达式在 docsIds 为空时不拼入 SQL
                .in(request.getDocsIds() != null && !request.getDocsIds().isEmpty(), AigcDocs::getId, request.getDocsIds())
                .orderByDesc(AigcDocs::getUpdateTime)
                .list();
    }

    private KnowledgeDocumentPreview buildPreview(AigcDocs docs, KnowledgeParsePreviewRequest request) {
        ParsedDocumentContent parsedDocument = documentContentParseService.parse(docs);
        DocumentSplitConfig splitConfig = resolveSplitConfig(docs, request.getChunkSize(), request.getOverlapSize());
        List<String> chunks = chunkDocument(parsedDocument, splitConfig.chunkSize(), splitConfig.overlapSize());
        int sectionLimit = sanitizeLimit(request.getSectionLimit(), DEFAULT_PREVIEW_SECTION_LIMIT);
        int chunkLimit = sanitizeLimit(request.getChunkLimit(), DEFAULT_PREVIEW_CHUNK_LIMIT);

        KnowledgeDocumentPreview preview = new KnowledgeDocumentPreview();
        preview.setDocsId(docs.getId());
        preview.setTitle(parsedDocument.getTitle());
        preview.setParserName(parsedDocument.getParserName());
        preview.setSectionCount(parsedDocument.getSections().size());
        preview.setChunkCount(chunks.size());
        preview.setContentLength(parsedDocument.getPlainText() == null ? 0 : parsedDocument.getPlainText().length());
        preview.setSections(limitList(parsedDocument.getSections(), sectionLimit));
        preview.setChunks(limitList(chunks, chunkLimit));
        log.info(
                "完成知识库文档解析预览，docsId={}, parserName={}, sectionCount={}, chunkCount={}, chunkSize={}, overlapSize={}",
                docs.getId(),
                parsedDocument.getParserName(),
                preview.getSectionCount(),
                preview.getChunkCount(),
                splitConfig.chunkSize(),
                splitConfig.overlapSize()
        );
        return preview;
    }

    private List<String> chunkDocument(ParsedDocumentContent parsedDocument, Integer chunkSize, Integer overlapSize) {
        List<String> chunks = new ArrayList<>();
        for (String section : parsedDocument.getSections()) {
            chunks.addAll(documentChunkingService.chunk(section, chunkSize, overlapSize));
        }
        return chunks;
    }

    private KnowledgeIndexRequest buildIndexRequest(KnowledgeParsePreviewRequest request) {
        KnowledgeIndexRequest indexRequest = new KnowledgeIndexRequest();
        indexRequest.setDocsIds(request.getDocsIds());
        indexRequest.setChunkSize(request.getChunkSize());
        indexRequest.setOverlapSize(request.getOverlapSize());
        return indexRequest;
    }

    /**
     * 解析文档切片配置，优先级三级递进：
     * 请求参数 → 文档 ingestionConfig（同时兼容 camelCase 与 snake_case 键名）→ 留空由处理器用内置默认。
     *
     * <p>注意：chunkSize 与 overlapSize 各自独立回退，可能一个来自请求、一个来自文档配置；
     * 两者均为 null 时处理器按默认切片策略执行。</p>
     */
    private DocumentSplitConfig resolveSplitConfig(AigcDocs docs, Integer requestChunkSize, Integer requestOverlapSize) {
        Integer chunkSize = requestChunkSize;
        Integer overlapSize = requestOverlapSize;
        // 快路径：请求参数两项都给了，直接使用，无需解析文档配置
        if (chunkSize != null && overlapSize != null) {
            return new DocumentSplitConfig(chunkSize, overlapSize);
        }
        // 解析文档级 ingestionConfig（解析失败返回空 Map，不阻断流程）
        Map<String, Object> ingestionConfig = parseIngestionConfig(docs);
        if (chunkSize == null) {
            // 兼容 chunkSize 与 chunk_size 两种键名写法
            chunkSize = integerValue(
                    ingestionConfig.get(CONFIG_CHUNK_SIZE),
                    ingestionConfig.get(CONFIG_CHUNK_SIZE_ALT)
            );
        }
        if (overlapSize == null) {
            // 兼容 overlapSize 与 overlap_size 两种键名写法
            overlapSize = integerValue(
                    ingestionConfig.get(CONFIG_OVERLAP_SIZE),
                    ingestionConfig.get(CONFIG_OVERLAP_SIZE_ALT)
            );
        }
        // 可能为 null/null 或部分 null，null 项交由处理器按默认值处理
        return new DocumentSplitConfig(chunkSize, overlapSize);
    }

    private Map<String, Object> parseIngestionConfig(AigcDocs docs) {
        if (docs.getIngestionConfig() == null || docs.getIngestionConfig().isBlank()) {
            return Map.of();
        }
        try {
            return objectMapper.readValue(docs.getIngestionConfig(), MAP_TYPE);
        } catch (Exception ex) {
            log.warn("解析文档 ingestionConfig 失败，docsId={}, ingestionConfig={}", docs.getId(), docs.getIngestionConfig(), ex);
            return Map.of();
        }
    }

    private Integer integerValue(Object... values) {
        for (Object value : values) {
            if (value instanceof Number number) {
                return number.intValue();
            }
            if (value instanceof String stringValue && !stringValue.isBlank()) {
                try {
                    return Integer.parseInt(stringValue.trim());
                } catch (NumberFormatException ignored) {
                    log.debug("忽略无法转换的切片配置值，value={}", stringValue);
                }
            }
        }
        return null;
    }

    private int sanitizeLimit(Integer limit, int defaultValue) {
        if (limit == null || limit <= 0) {
            return defaultValue;
        }
        return Math.min(limit, 50);
    }

    private List<String> limitList(List<String> values, int limit) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        return values.stream().limit(limit).toList();
    }

    /**
     * 将文档标记为「排队中」并清空历史状态，确保重复索引/重试从干净起点开始。
     *
     * <p>关键点：同时重置索引状态（indexingStatus，DB 整数状态）与向量化状态
     * （embedStatus，字符串编码），并清空错误信息与开始/结束时间——
     * 否则上一次的失败痕迹会在重新排队时被误报给前端。
     * 内存对象与数据库双写，保证同事务内后续读取一致。</p>
     */
    private void markDocsQueued(AigcDocs docs) {
        long now = System.currentTimeMillis();
        // 原地更新内存对象，避免外层循环再取一次库
        docs.setIndexingStatus(DocumentIndexStatusEnum.PENDING.dbStatus());
        docs.setEmbedStatus(DocumentIndexStatusEnum.PENDING.code());
        docs.setEmbedError(null);
        docs.setEmbedStartTime(null);
        docs.setEmbedEndTime(null);
        docs.setUpdateTime(now);
        // 持久化到 DB，前端轮询 queryIndexStatus 时立即看到「排队中」
        aigcDocsService.lambdaUpdate()
                .eq(AigcDocs::getId, docs.getId())
                .set(AigcDocs::getIndexingStatus, DocumentIndexStatusEnum.PENDING.dbStatus())
                .set(AigcDocs::getEmbedStatus, DocumentIndexStatusEnum.PENDING.code())
                .set(AigcDocs::getEmbedError, null)
                .set(AigcDocs::getEmbedStartTime, null)
                .set(AigcDocs::getEmbedEndTime, null)
                .set(AigcDocs::getUpdateTime, now)
                .update();
    }

    private Map<String, SegmentSummary> loadSegmentSummary(List<AigcDocs> docsList) {
        List<String> docsIds = docsList.stream().map(AigcDocs::getId).toList();
        Map<String, SegmentSummary> summary = new HashMap<>();
        for (Map<String, Object> row : aigcSegmentService.summarizeByDocsIds(docsIds)) {
            Object docsId = row.get("docsId");
            if (docsId == null) {
                continue;
            }
            summary.put(docsId.toString(), new SegmentSummary(
                    intValue(row.get("segmentCount")),
                    longValue(row.get("charCount"))
            ));
        }
        return summary;
    }

    private int intValue(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        return 0;
    }

    private long longValue(Object value) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        return 0L;
    }

    private KnowledgeDocumentIndexStatus buildDocumentIndexStatus(
            AigcDocs docs,
            long now,
            Map<String, SegmentSummary> segmentSummary
    ) {
        SegmentSummary summary = segmentSummary.getOrDefault(docs.getId(), EMPTY_SEGMENT_SUMMARY);
        KnowledgeDocumentIndexStatus status = new KnowledgeDocumentIndexStatus();
        status.setDocsId(docs.getId());
        status.setName(docs.getName());
        status.setExt(docs.getExt());
        status.setIndexingStatus(docs.getIndexingStatus());
        String embedStatus = resolveEmbedStatus(docs);
        status.setEmbedStatus(embedStatus);
        status.setEmbedError(Objects.equals(embedStatus, DocumentIndexStatusEnum.FAILED.code())
                ? docs.getEmbedError()
                : null);
        status.setEmbedStartTime(docs.getEmbedStartTime());
        status.setEmbedEndTime(docs.getEmbedEndTime());
        status.setCostMs(calculateCostMs(docs, now));
        status.setSegmentCount(summary.segmentCount());
        status.setCharCount(summary.charCount());
        status.setUpdateTime(docs.getUpdateTime());
        return status;
    }

    private String resolveEmbedStatus(AigcDocs docs) {
        if (docs.getEmbedStatus() != null && !docs.getEmbedStatus().isBlank()) {
            return docs.getEmbedStatus();
        }
        if (Objects.equals(docs.getIndexingStatus(), DocumentIndexStatusEnum.RUNNING.dbStatus())) {
            return DocumentIndexStatusEnum.RUNNING.code();
        }
        if (Objects.equals(docs.getIndexingStatus(), DocumentIndexStatusEnum.COMPLETED.dbStatus())) {
            return DocumentIndexStatusEnum.COMPLETED.code();
        }
        if (Objects.equals(docs.getIndexingStatus(), DocumentIndexStatusEnum.FAILED.dbStatus())) {
            return DocumentIndexStatusEnum.FAILED.code();
        }
        return DocumentIndexStatusEnum.PENDING.code();
    }

    private Long calculateCostMs(AigcDocs docs, long now) {
        if (docs.getEmbedStartTime() == null) {
            return null;
        }
        long endTime = docs.getEmbedEndTime() == null ? now : docs.getEmbedEndTime();
        return Math.max(0L, endTime - docs.getEmbedStartTime());
    }

    private int countByStatus(List<KnowledgeDocumentIndexStatus> docs, String status) {
        return (int) docs.stream()
                .filter(item -> Objects.equals(status, item.getEmbedStatus()))
                .count();
    }

    private int calculateProgressPercent(int total, int completedCount, int failedCount) {
        if (total <= 0) {
            return 100;
        }
        return Math.min(100, (completedCount + failedCount) * 100 / total);
    }

    /**
     * 文档切片配置。
     *
     * @param chunkSize 切片大小
     * @param overlapSize 重叠大小
     * @author LangChat Team
     * @since 2026/3/25
     */
    private record DocumentSplitConfig(
            Integer chunkSize,
            Integer overlapSize
    ) {
    }

    /**
     * 文档分段统计。
     *
     * @param segmentCount 分段条数
     * @param charCount 分段字符数
     * @author LangChat Team
     * @since 2026/9/27
     */
    private record SegmentSummary(
            int segmentCount,
            long charCount
    ) {
    }
}
