package cn.langchat.core.runtime.config;

/**
 * 知识库索引线程池名称常量。
 *
 * @author LangChat Team
 * @since 2026/3/25
 */
public final class KnowledgeIndexExecutorNames {

    /** 文档向量化异步线程池。 */
    public static final String DOCUMENT_INDEX_TASK_EXECUTOR = "documentIndexTaskExecutor";

    /** 私有构造：常量类，禁止实例化。 */
    private KnowledgeIndexExecutorNames() {
    }
}
