package cn.langchat.core.runtime.skill;

import cn.langchat.aigc.biz.entity.AigcSkill;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.service.tool.ToolExecutor;

/**
 * Agent Skill 注册中心。
 *
 * @author LangChat Team
 * @since 2026/3/24
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AgentToolRegistry {

    private final List<AgentToolProvider> providers;

    /**
     * 按 Skill 配置构建可用工具列表。
     */
    public Map<ToolSpecification, ToolExecutor> buildTools(List<AigcSkill> skills, AgentToolContext baseContext) {
        Map<ToolSpecification, ToolExecutor> tools = new LinkedHashMap<>();
        for (AigcSkill skill : skills) {
            // 找到第一个支持该 Skill 的提供器，用「当前 skill + 会话/用户/知识库」构建独立工具上下文
            providers.stream()
                    .filter(provider -> provider.supports(skill))
                    .findFirst()
                    .map(provider -> provider.buildDefinition(new AgentToolContext(
                            baseContext.agent(),
                            baseContext.conversationId(),
                            baseContext.userId(),
                            skill,
                            baseContext.knowledges()
                    )))
                    .ifPresentOrElse(
                            // 命中：注册工具规格与执行器，供模型按需调用
                            definition -> tools.put(definition.specification(), definition.executor()),
                            // 未命中：仅告警跳过，不中断整体构建
                            () -> log.warn("未找到 Skill 提供器，skillName={}", skill.getName())
                    );
        }
        return tools;
    }
}
