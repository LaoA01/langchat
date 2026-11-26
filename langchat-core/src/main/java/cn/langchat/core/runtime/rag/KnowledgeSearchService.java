package cn.langchat.core.runtime.rag;

import cn.langchat.aigc.biz.entity.AigcDocs;
import cn.langchat.aigc.biz.entity.AigcKnowledge;
import cn.langchat.aigc.biz.entity.AigcSegment;
import cn.langchat.aigc.biz.service.AigcDocsService;
import cn.langchat.aigc.biz.service.AigcSegmentService;
import cn.langchat.core.chat.model.knowledge.KnowledgeSearchHit;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 知识检索服务。
 *
 * @author LangChat Team
 * @since 2026/3/25
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class KnowledgeSearchService {

    private static final int DEFAULT_TOP_K = 5;
    private static final int DEFAULT_FETCH_MULTIPLIER = 6;

    private final AigcSegmentService aigcSegmentService;
    private final AigcDocsService aigcDocsService;

    /**
     * 执行基于切片表的关键词检索。
     */
    public List<KnowledgeSearchHit> search(List<AigcKnowledge> knowledges, String query, Integer topK) {
        if (knowledges == null || knowledges.isEmpty() || query == null || query.isBlank()) {
            return List.of();
        }
        List<String> knowledgeIds = knowledges.stream()
                .map(AigcKnowledge::getId)
                .filter(Objects::nonNull)
                .distinct()
                .toList();
        if (knowledgeIds.isEmpty()) {
            return List.of();
        }

        int limitedTopK = normalizeTopK(topK);
        Page<AigcSegment> page = new Page<>(1, (long) limitedTopK * DEFAULT_FETCH_MULTIPLIER);
        List<AigcSegment> segments = aigcSegmentService.lambdaQuery()
                .in(AigcSegment::getKnowledgeId, knowledgeIds)
                .eq(AigcSegment::getEnabled, Boolean.TRUE)
                .like(AigcSegment::getContent, query)
                .page(page)
                .getRecords();
        if (segments.isEmpty()) {
            return List.of();
        }

        Map<String, AigcKnowledge> knowledgeMap = knowledges.stream()
                .collect(Collectors.toMap(AigcKnowledge::getId, Function.identity(), (left, right) -> left));
        Map<String, AigcDocs> docsMap = loadDocsMap(segments);

        List<KnowledgeSearchHit> hits = segments.stream()
                .map(segment -> toHit(segment, query, knowledgeMap, docsMap))
                .sorted((left, right) -> Double.compare(right.getScore(), left.getScore()))
                .limit(limitedTopK)
                .toList();
        log.info("完成关键词知识检索，knowledgeSize={}, query={}, hitSize={}", knowledges.size(), query, hits.size());
        return hits;
    }

    /**
     * 批量按文档 ID 加载文档记录并建立索引映射，避免命名回填时的 N+1 查询。
     */
    private Map<String, AigcDocs> loadDocsMap(List<AigcSegment> segments) {
        List<String> docsIds = segments.stream()
                .map(AigcSegment::getDocsId)
                .filter(Objects::nonNull)
                .distinct()
                .toList();
        if (docsIds.isEmpty()) {
            return Collections.emptyMap();
        }
        return aigcDocsService.lambdaQuery()
                .in(AigcDocs::getId, docsIds)
                .list()
                .stream()
                .collect(Collectors.toMap(AigcDocs::getId, Function.identity(), (left, right) -> left));
    }

    /**
     * 将切片记录转换为检索命中结果：按评分公式计算相似度得分，
     * 并回填知识库名称、文档名称、截断正文与元数据。
     */
    private KnowledgeSearchHit toHit(
            AigcSegment segment,
            String query,
            Map<String, AigcKnowledge> knowledgeMap,
            Map<String, AigcDocs> docsMap
    ) {
        AigcKnowledge knowledge = knowledgeMap.get(segment.getKnowledgeId());
        AigcDocs docs = docsMap.get(segment.getDocsId());
        double score = calculateScore(segment.getContent(), query, segment.getPosition());

        KnowledgeSearchHit hit = new KnowledgeSearchHit();
        hit.setKnowledgeId(segment.getKnowledgeId());
        hit.setKnowledgeName(knowledge == null ? null : knowledge.getName());
        hit.setDocsId(segment.getDocsId());
        hit.setDocsName(docs == null ? null : docs.getName());
        hit.setSegmentId(segment.getId());
        hit.setScore(score);
        hit.setContent(trimContent(segment.getContent()));
        hit.setMetadata(buildMetadata(segment, knowledge, docs, score));
        return hit;
    }

    /**
     * 构建命中元数据：知识库、文档、分段标识 + 位置与得分，供前端渲染引用来源。
     */
    private Map<String, Object> buildMetadata(AigcSegment segment, AigcKnowledge knowledge, AigcDocs docs, double score) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        putIfPresent(metadata, "knowledgeId", segment.getKnowledgeId());
        putIfPresent(metadata, "knowledgeName", knowledge == null ? null : knowledge.getName());
        putIfPresent(metadata, "docsId", segment.getDocsId());
        putIfPresent(metadata, "docsName", docs == null ? null : docs.getName());
        putIfPresent(metadata, "segmentId", segment.getId());
        metadata.put("position", segment.getPosition());
        metadata.put("score", score);
        putIfPresent(metadata, "segmentName", segment.getName());
        return metadata;
    }

    /**
     * 简易关键词评分：词频×2 + 首次命中加分 + 分段位置权重（越靠前加分越多）。
     */
    private double calculateScore(String content, String query, Integer position) {
        String safeContent = content == null ? "" : content.toLowerCase();
        String safeQuery = query.toLowerCase();
        int frequency = countOccurrences(safeContent, safeQuery);
        int firstIndex = safeContent.indexOf(safeQuery);
        double exactMatchBonus = firstIndex >= 0 ? 1.0D : 0.0D;
        double positionBonus = position == null ? 0.0D : Math.max(0.0D, 1.0D - (position * 0.01D));
        return frequency * 2.0D + exactMatchBonus + positionBonus;
    }

    /** 统计查询词在内容中完整出现的次数。 */
    private int countOccurrences(String content, String query) {
        if (query.isBlank()) {
            return 0;
        }
        int count = 0;
        int fromIndex = 0;
        while (fromIndex >= 0) {
            fromIndex = content.indexOf(query, fromIndex);
            if (fromIndex >= 0) {
                count++;
                fromIndex += query.length();
            }
        }
        return count;
    }

    /** 命中正文最多保留 300 字符，超长部分截断。 */
    private String trimContent(String content) {
        if (content == null) {
            return "";
        }
        return content.length() <= 300 ? content : content.substring(0, 300);
    }

    /** 归一化 topK：未配置或非正值时回退默认值 5。 */
    private int normalizeTopK(Integer topK) {
        return topK == null || topK <= 0 ? DEFAULT_TOP_K : topK;
    }

    /** 值非 null 时才写入元数据，避免出现 null 字段。 */
    private void putIfPresent(Map<String, Object> metadata, String key, Object value) {
        if (value != null) {
            metadata.put(key, value);
        }
    }
}
