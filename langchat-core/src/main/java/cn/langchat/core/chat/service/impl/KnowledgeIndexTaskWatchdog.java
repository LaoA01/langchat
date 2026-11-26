package cn.langchat.core.chat.service.impl;

import cn.langchat.aigc.biz.entity.AigcDocs;
import cn.langchat.aigc.biz.service.AigcDocsService;
import cn.langchat.core.chat.enums.DocumentIndexStatusEnum;
import cn.langchat.core.runtime.config.KnowledgeIndexAsyncProperties;
import java.time.Duration;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * 知识库向量化任务状态守护器。
 *
 * <p>定期收敛因服务中断、线程异常等原因长期停留在等待中/执行中的任务，
 * 同时清理历史完成记录中残留的错误信息。</p>
 *
 * @author LangChat Team
 * @since 2026/9/27
 */
@Component
@RequiredArgsConstructor
@Slf4j
@EnableScheduling
public class KnowledgeIndexTaskWatchdog {

    /** 超时收敛时写入文档的统一错误提示。 */
    private static final String TIMEOUT_ERROR =
            "向量化任务执行超时，可能因服务中断或任务阻塞，已自动标记为失败";

    private final AigcDocsService aigcDocsService;
    private final KnowledgeIndexAsyncProperties properties;

    /**
     * 后台扫描所有尚未结束的任务。
     */
    @Scheduled(fixedDelayString = "#{@knowledgeIndexAsyncProperties.staleTaskSweepInterval.toMillis()}")
    /**
     * 周期性扫描所有未结束的向量化任务：
     * 查出处于 PENDING/RUNNING 的文档（含仅旧 indexingStatus 字段的遗留数据），
     * 逐个判定超时并标记失败；随后清理已完成记录中残留的错误信息。
     */
    public void expireStaleTasks() {
        // 查询未结束文档：新字段 embedStatus 为 PENDING/RUNNING，或遗留数据 embedStatus 为空但 indexingStatus 为激活态
        List<AigcDocs> activeDocs = aigcDocsService.lambdaQuery()
                .and(status -> status
                        .in(AigcDocs::getEmbedStatus, activeStatusCodes())
                        .or()
                        .and(legacy -> legacy
                                .isNull(AigcDocs::getEmbedStatus)
                                .in(AigcDocs::getIndexingStatus, activeDbStatuses())))
                .list();
        long now = System.currentTimeMillis();
        int expiredCount = 0;
        // 逐个判定超时并标记失败，统计收敛数量
        for (AigcDocs docs : activeDocs) {
            if (expireIfStale(docs, now)) {
                expiredCount++;
            }
        }
        // 已完成但残留错误信息的文档，清理错误字段保持数据整洁
        aigcDocsService.lambdaQuery()
                .eq(AigcDocs::getEmbedStatus, DocumentIndexStatusEnum.COMPLETED.code())
                .isNotNull(AigcDocs::getEmbedError)
                .list()
                .forEach(docs -> clearCompletedError(docs, now));
        if (expiredCount > 0) {
            log.warn("知识库向量化超时任务已收敛，expiredCount={}", expiredCount);
        }
    }

    /** 将超时文档更新为失败，返回是否实际完成更新。 */
    /**
     * 若文档仍处于激活态且已超过任务超时阈值，则原子更新为失败状态
     * （条件更新旧状态防竞态），并同步内存对象与日志告警。
     *
     * @return 是否实际完成更新（用于收敛计数）
     */
    private boolean expireIfStale(AigcDocs docs, long now) {
        // 非激活或未超时不处理
        if (!isActive(docs) || !isExpired(docs, now)) {
            return false;
        }
        // 条件更新：WHERE 状态仍为激活态才置为失败，避免覆盖并发更新（乐观锁语义）
        boolean updated = aigcDocsService.lambdaUpdate()
                .eq(AigcDocs::getId, docs.getId())
                .and(status -> status
                        .in(AigcDocs::getEmbedStatus, activeStatusCodes())
                        .or()
                        .and(legacy -> legacy
                                .isNull(AigcDocs::getEmbedStatus)
                                .in(AigcDocs::getIndexingStatus, activeDbStatuses())))
                .set(AigcDocs::getIndexingStatus, DocumentIndexStatusEnum.FAILED.dbStatus())
                .set(AigcDocs::getEmbedStatus, DocumentIndexStatusEnum.FAILED.code())
                .set(AigcDocs::getEmbedError, TIMEOUT_ERROR)
                .set(AigcDocs::getEmbedEndTime, now)
                .set(AigcDocs::getUpdateTime, now)
                .update();
        if (updated) {
            docs.setIndexingStatus(DocumentIndexStatusEnum.FAILED.dbStatus());
            docs.setEmbedStatus(DocumentIndexStatusEnum.FAILED.code());
            docs.setEmbedError(TIMEOUT_ERROR);
            docs.setEmbedEndTime(now);
            docs.setUpdateTime(now);
            log.warn("知识库文档向量化任务超时，docsId={}, startTime={}", docs.getId(), taskStartTime(docs));
        }
        return updated;
    }

    /** 清理已完成文档残留的错误信息：条件更新（状态必须仍为 COMPLETED）确保不覆盖并发变更。 */
    private void clearCompletedError(AigcDocs docs, long now) {
        if (docs.getEmbedError() == null || docs.getEmbedError().isBlank()) {
            return;
        }
        // 仅当状态仍为 COMPLETED 时清空错误字段，防止与并发操作冲突
        boolean updated = aigcDocsService.lambdaUpdate()
                .eq(AigcDocs::getId, docs.getId())
                .eq(AigcDocs::getEmbedStatus, DocumentIndexStatusEnum.COMPLETED.code())
                .set(AigcDocs::getEmbedError, null)
                .set(AigcDocs::getUpdateTime, now)
                .update();
        if (updated) {
            docs.setEmbedError(null);
            docs.setUpdateTime(now);
        }
    }

    /** 判断文档是否处于未结束（PENDING/RUNNING）：优先看新字段 embedStatus，为空时回退旧字段 indexingStatus。 */
    private boolean isActive(AigcDocs docs) {
        if (docs.getEmbedStatus() != null && !docs.getEmbedStatus().isBlank()) {
            return activeStatusCodes().contains(docs.getEmbedStatus());
        }
        return activeDbStatuses().contains(docs.getIndexingStatus());
    }

    /**
     * 判断任务是否超时：配置 taskTimeout 为空/零/负值视为不启用守护（永远不超时）；
     * 有配置时按任务开始时间与当前时间差比较。
     */
    private boolean isExpired(AigcDocs docs, long now) {
        Duration timeout = properties.getTaskTimeout();
        // 未配置超时或配置非法则不做守护，避免误标正常任务
        if (timeout == null || timeout.isZero() || timeout.isNegative()) {
            return false;
        }
        Long startTime = taskStartTime(docs);
        return startTime != null && now - startTime >= timeout.toMillis();
    }

    /** 取任务最可用的开始时间：优先 embedStartTime，其次 updateTime，最后 createTime。 */
    private Long taskStartTime(AigcDocs docs) {
        if (docs.getEmbedStartTime() != null) {
            return docs.getEmbedStartTime();
        }
        if (docs.getUpdateTime() != null) {
            return docs.getUpdateTime();
        }
        return docs.getCreateTime();
    }

    /** 新字段 embedStatus 的激活值集合（PENDING/RUNNING 字符串码）。 */
    private List<String> activeStatusCodes() {
        return List.of(
                DocumentIndexStatusEnum.PENDING.code(),
                DocumentIndexStatusEnum.RUNNING.code()
        );
    }

    /** 旧字段 indexingStatus 的激活值集合（PENDING/RUNNING 数字码）。 */
    private List<Integer> activeDbStatuses() {
        return List.of(
                DocumentIndexStatusEnum.PENDING.dbStatus(),
                DocumentIndexStatusEnum.RUNNING.dbStatus()
        );
    }
}
