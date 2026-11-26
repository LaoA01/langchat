package cn.langchat.core.runtime;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

/**
 * 智能问数 AI Service。
 */
public interface DataAnalysisAiService {

    /**
     * 发起智能问数流式分析：按系统提示词对用户自然语言问题进行数据查询与分析。
     */
    @SystemMessage("{{systemPrompt}}")
    TokenStream analyze(
            @V("systemPrompt") String systemPrompt,
            @UserMessage String question
    );
}
