package cn.langchat.core.runtime.rag;

import cn.langchat.aigc.biz.entity.AigcKnowledge;
import cn.langchat.core.chat.model.knowledge.KnowledgeSearchHit;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import java.util.List;

/**
 * 基于切片表的关键词检索器。
 *
 * @author LangChat Team
 * @since 2026/3/25
 */
public class KeywordSegmentContentRetriever implements ContentRetriever {

    private final KnowledgeSearchService knowledgeSearchService;
    private final List<AigcKnowledge> knowledges;
    private final int topK;

    /**
     * 构造关键词检索器。
     *
     * @param knowledgeSearchService 知识检索服务，负责底层切片表检索
     * @param knowledges 本次对话绑定的知识库列表（空列表时检索无结果）
     * @param topK 每次检索召回的最大条数
     */
    public KeywordSegmentContentRetriever(
            KnowledgeSearchService knowledgeSearchService,
            List<AigcKnowledge> knowledges,
            int topK
    ) {
        this.knowledgeSearchService = knowledgeSearchService;
        this.knowledges = knowledges;
        this.topK = topK;
    }

    /**
     * 执行关键词检索：按用户问题在绑定知识库的切片中匹配，转换为 LangChain4j Content。
     */
    @Override
    public List<Content> retrieve(Query query) {
        return knowledgeSearchService.search(knowledges, query.text(), topK).stream()
                .map(this::toContent)
                .toList();
    }

    /** 将知识库命中结果转换为 LangChain4j Content：正文作为文本段、元数据原样透传。 */
    private Content toContent(KnowledgeSearchHit hit) {
        Metadata metadata = Metadata.from(hit.getMetadata() == null ? java.util.Map.of() : hit.getMetadata());
        return Content.from(TextSegment.from(hit.getContent(), metadata));
    }
}
