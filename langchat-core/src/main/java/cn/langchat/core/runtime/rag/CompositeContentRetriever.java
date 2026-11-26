package cn.langchat.core.runtime.rag;

import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * 多知识库聚合检索器。
 *
 * @author LangChat Team
 * @since 2026/3/24
 */
public class CompositeContentRetriever implements ContentRetriever {

    private final List<ContentRetriever> retrievers;
    private final int maxResults;

    /**
     * 构造聚合检索器。
     *
     * @param retrievers 子检索器列表（关键词检索器与各知识库向量检索器）
     * @param maxResults 合并去重后整体召回的最大条数
     */
    public CompositeContentRetriever(List<ContentRetriever> retrievers, int maxResults) {
        this.retrievers = retrievers;
        this.maxResults = maxResults;
    }

    /**
     * 依次调用各子检索器召回内容，按文本去重合并，达到 maxResults 上限即停止。
     */
    @Override
    public List<Content> retrieve(Query query) {
        LinkedHashMap<String, Content> merged = new LinkedHashMap<>();
        for (ContentRetriever retriever : retrievers) {
            List<Content> contents = retriever.retrieve(query);
            for (Content content : contents) {
                merged.putIfAbsent(content.textSegment().text(), content);
                if (merged.size() >= maxResults) {
                    return new ArrayList<>(merged.values());
                }
            }
        }
        return new ArrayList<>(merged.values());
    }
}
