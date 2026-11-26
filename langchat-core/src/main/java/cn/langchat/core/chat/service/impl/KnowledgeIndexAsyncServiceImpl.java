package cn.langchat.core.chat.service.impl;

import cn.langchat.core.chat.service.KnowledgeDocumentIndexProcessor;
import cn.langchat.core.chat.service.KnowledgeIndexAsyncService;
import cn.langchat.core.runtime.config.KnowledgeIndexExecutorNames;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

/**
 * 知识库异步索引服务实现。
 *
 * @author LangChat Team
 * @since 2026/3/25
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class KnowledgeIndexAsyncServiceImpl implements KnowledgeIndexAsyncService {

    private final KnowledgeDocumentIndexProcessor knowledgeDocumentIndexProcessor;

    /**
     * 将单文档索引任务派发到专用异步线程池执行。
     *
     * <p>由 {@code @Async(DOCUMENT_INDEX_TASK_EXECUTOR)} 切换到 documentIndexTaskExecutor 线程池，
     * 本方法对调用方立即返回，实际的解析→切片→向量化→写入向量库工作由
     * {@link KnowledgeDocumentIndexProcessor#processDocument} 在后台完成。
     * 线程池配置了 CallerRunsPolicy：队列打满时退化为调用线程执行，任务不丢失。</p>
     *
     * @param knowledgeId 知识库 ID
     * @param docsId 文档 ID
     * @param chunkSize 切片大小，null 时由处理器按默认策略
     * @param overlapSize 切片重叠大小，null 时由处理器按默认策略
     */
    @Override
    @Async(KnowledgeIndexExecutorNames.DOCUMENT_INDEX_TASK_EXECUTOR)
    public void submitDocumentIndex(String knowledgeId, String docsId, Integer chunkSize, Integer overlapSize) {
        log.info("提交文档异步向量化任务，knowledgeId={}, docsId={}", knowledgeId, docsId);
        knowledgeDocumentIndexProcessor.processDocument(knowledgeId, docsId, chunkSize, overlapSize);
    }
}
