package cn.langchat.core.runtime.skill.impl;

import cn.langchat.aigc.biz.entity.AigcSkill;
import cn.langchat.aigc.biz.service.SkillPackageService;
import cn.langchat.core.runtime.skill.AgentToolContext;
import cn.langchat.core.runtime.skill.AgentToolDefinition;
import cn.langchat.core.runtime.skill.AgentToolProvider;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import java.util.regex.Pattern;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

/**
 * 标准技能包工具提供器。
 *
 * <p>技能包以 SKILL.md 为入口（frontmatter 声明 name/description，正文为标准化的技能指令文档），
 * 运行时基于 LangChain4j 官方 Tool 机制动态注册：把每个启用的技能暴露为一个工具，
 * 模型调用工具后返回技能指令文档（可携带 path 读取包内 references 等附加文档），
 * 由模型按照文档指令继续执行，实现技能文档的动态调用。</p>
 *
 * @author LangChat Team
 * @since 2026/9/26
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DocumentSkillToolProvider implements AgentToolProvider {

    /** 工具名中不允许出现的字符（非字母/数字/下划线/连字符），用于把技能名清洗为合法工具名。 */
    private static final Pattern INVALID_TOOL_NAME = Pattern.compile("[^a-zA-Z0-9_-]+");

    /** 技能包服务：负责技能工作区准备与文档读取。 */
    private final SkillPackageService skillPackageService;
    /** JSON 解析器，用于解析工具入参中的 path 字段。 */
    private final ObjectMapper objectMapper;

    /**
     * 作为标准技能包的默认提供器，支持所有技能；
     * 注册侧按 supports 过滤匹配，后续若有更专属的提供器可优先命中。
     */
    @Override
    public boolean supports(AigcSkill skill) {
        return true;
    }

    /**
     * 构建单个技能的工具定义：由工具规格（名称/描述/参数）与执行回调组成，
     * 模型发起工具调用时触发回调，读取技能指令文档并返回给模型按文档执行。
     */
    @Override
    public AgentToolDefinition buildDefinition(AgentToolContext context) {
        // 从上下文取出当前技能，工具名/描述/参数均基于该技能生成
        AigcSkill skill = context.skill();
        ToolSpecification specification = ToolSpecification.builder()
                .name(resolveToolName(skill))
                .description(resolveDescription(skill))
                .parameters(buildParameters())
                .build();
        // 执行回调绑定当前技能；memoryId 未使用（技能文档读取与会话无关）
        return new AgentToolDefinition(specification, (request, memoryId) -> execute(request, skill));
    }

    /**
     * 工具执行入口：按可选 path 读取技能包内文档并返回文本。
     * path 为空读 SKILL.md 主文档，传目录返回文件清单，传文件返回正文。
     * 读取失败不抛异常，返回提示文本交由模型处理，避免中断对话链路。
     */
    private String execute(ToolExecutionRequest request, AigcSkill skill) {
        // 从工具入参解析可选 path（为空表示读取主文档）
        String path = extractPath(request);
        log.info("执行技能文档工具，toolName={}, skillName={}, path={}", request.name(), skill.getName(), path);
        try {
            // 委托技能包服务读取文档（内部含工作区准备与路径安全校验）
            return skillPackageService.readSkillDocument(skill, path);
        } catch (Exception exception) {
            // 读取失败降级为提示文本，不让对话因工具异常而中断
            log.warn("技能文档读取失败，skillName={}, path={}", skill.getName(), path, exception);
            return "技能文档读取失败：" + exception.getMessage();
        }
    }

    /**
     * 从工具入参 JSON 中提取可选的 path 字段。
     * 入参为空、非合法 JSON 或未携带 path 时均返回 null（等价于读取主文档）。
     */
    private String extractPath(ToolExecutionRequest request) {
        String arguments = request.arguments();
        // 无入参直接返回 null，表示读取 SKILL.md 主文档
        if (!StringUtils.hasText(arguments)) {
            return null;
        }
        try {
            // 解析入参 JSON 并取出 path 字段，空串同样回退为 null
            JsonNode node = objectMapper.readTree(arguments.trim());
            String path = node.path("path").asText(null);
            return StringUtils.hasText(path) ? path : null;
        } catch (Exception exception) {
            // 解析失败仅告警并回退主文档，不阻断工具执行
            log.warn("解析技能工具入参失败，arguments={}", arguments);
            return null;
        }
    }

    /**
     * 将技能名（无则用 id）清洗为模型可接受的合法工具名：
     * 转小写 → 非法字符替换为连字符 → 去首尾连字符 → 兜底防空 → 截断至 64 字符。
     */
    private String resolveToolName(AigcSkill skill) {
        // 优先使用技能名，缺失时回退技能 id
        String name = StringUtils.hasText(skill.getName()) ? skill.getName() : skill.getId();
        // 转小写，并将非法字符（非字母数字下划线连字符）统一替换为连字符
        String sanitized = INVALID_TOOL_NAME.matcher(name.toLowerCase()).replaceAll("-");
        // 去除首部连字符，避免工具名以连字符开头
        while (sanitized.startsWith("-")) {
            sanitized = sanitized.substring(1);
        }
        // 去除尾部连字符，避免工具名以连字符结尾
        while (sanitized.endsWith("-")) {
            sanitized = sanitized.substring(0, sanitized.length() - 1);
        }
        // 清洗后为空时用 skill-{id} 兜底，保证工具名非空
        if (!StringUtils.hasText(sanitized)) {
            sanitized = "skill-" + (skill.getId() == null ? "unknown" : skill.getId());
        }
        // 工具名长度上限 64，超长截断
        return sanitized.length() > 64 ? sanitized.substring(0, 64) : sanitized;
    }

    /**
     * 组装工具描述：标题 + 技能描述 + 调用指引，引导模型调用后按返回的指令文档执行；
     * 部分模型对工具描述长度敏感，超长时截断至 1024 字符。
     */
    private String resolveDescription(AigcSkill skill) {
        // 标题缺失时回退技能名
        String title = StringUtils.hasText(skill.getTitle()) ? skill.getTitle() : skill.getName();
        String description = StringUtils.hasText(skill.getDescription()) ? skill.getDescription() : "";
        // 固定格式说明：调用后返回标准指令文档，引导模型按文档执行
        String text = "技能文档：" + title + (description.isBlank() ? "" : " —— " + description)
                + "。调用后将返回该技能的标准指令文档，请按文档内容执行。";
        // 超过 1024 字符截断至 1021 并补省略号，满足模型工具描述长度限制
        return text.length() > 1024 ? text.substring(0, 1021) + "..." : text;
    }

    /**
     * 定义工具参数 schema：仅一个可选 path 字段，无必填项。
     * path 为空读主文档，传目录返回文件清单，传具体文件返回正文。
     */
    private JsonObjectSchema buildParameters() {
        return JsonObjectSchema.builder()
                .description("技能文档读取参数")
                // path 为可选参数；模型可用它读取技能包内的附加文档（如 references）
                .addStringProperty("path",
                        "可选：技能包内相对路径（如 references/api.md）。留空返回 SKILL.md 主文档，传目录路径返回文件清单")
                // 无必填参数，允许模型不带任何入参直接调用
                .required(java.util.List.of())
                // 允许额外属性，兼容部分模型对未知字段的容错
                .additionalProperties(Boolean.TRUE)
                .build();
    }
}
