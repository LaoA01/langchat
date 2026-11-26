package cn.langchat.aigc.biz.service.impl;

import cn.langchat.aigc.biz.config.SkillProperties;
import cn.langchat.aigc.biz.entity.AigcSkill;
import cn.langchat.aigc.biz.service.AigcSkillService;
import cn.langchat.aigc.biz.service.SkillPackageService;
import cn.langchat.aigc.biz.service.skill.SkillFileContent;
import cn.langchat.aigc.biz.service.skill.SkillFileNode;
import cn.langchat.aigc.biz.service.skill.SkillFrontmatter;
import cn.langchat.common.exception.BizException;
import cn.langchat.common.oss.model.OssObject;
import cn.langchat.common.oss.model.OssUploadRequest;
import cn.langchat.common.oss.service.OssService;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

/**
 * 技能包服务默认实现。
 *
 * @author LangChat Team
 * @since 2026/9/26
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SkillPackageServiceImpl implements SkillPackageService {

    /** 技能包默认入口主文档名。 */
    private static final String ENTRY_FILE = "SKILL.md";
    /** 工作区本地路径前缀。 */
    private static final String WORKSPACE_PREFIX = "skills/";
    /** 技能包上传/下载大小上限（500MB）。 */
    private static final long MAX_PACKAGE_SIZE = 500L * 1024 * 1024;
    private static final List<String> TEXT_EXTENSIONS = List.of(
            "md", "markdown", "txt", "json", "yaml", "yml", "xml", "html", "htm", "css",
            "js", "mjs", "cjs", "ts", "tsx", "jsx", "sh", "bash", "py", "java", "sql",
            "vue", "svelte", "go", "rs", "c", "h", "cc", "cpp", "hpp", "cs", "kt", "kts",
            "gradle", "groovy", "php", "rb", "swift", "scala", "lua", "r", "dart", "proto",
            "toml", "ini", "cfg", "conf", "csv", "properties", "env", "gitignore", "dockerfile");

    private final SkillProperties skillProperties;
    private final OssService ossService;
    private final AigcSkillService aigcSkillService;

    @Override
    public AigcSkill installPackage(byte[] zipBytes, String originalFilename, String tags) {
        if (zipBytes == null || zipBytes.length == 0) {
            throw new BizException("SKILL_PACKAGE_EMPTY", "技能包内容为空");
        }
        if (zipBytes.length > MAX_PACKAGE_SIZE) {
            throw new BizException("SKILL_PACKAGE_TOO_LARGE", "技能包超过大小限制（500MB）");
        }
        Map<String, byte[]> entries = unzip(zipBytes);
        String base = locateSkillRoot(entries);

        SkillFrontmatter frontmatter = parseEntryDocument(entries, base);
        String name = sanitizeSkillName(frontmatter.getName(), null);
        if (!StringUtils.hasText(name)) {
            throw new BizException(
                    "SKILL_FRONTMATTER_INVALID",
                    "SKILL.md 缺少 frontmatter name 字段，请检查技能包格式");
        }

        AigcSkill skill = new AigcSkill();
        skill.setName(name);
        skill.setTitle(firstNonBlank(frontmatter.getTitle(), name));
        skill.setDescription(firstNonBlank(frontmatter.getDescription(), "未填写技能描述"));
        skill.setVersion(firstNonBlank(frontmatter.getVersion(), "0.0.1"));
        skill.setLicense(frontmatter.getLicense());
        skill.setEntryFile(ENTRY_FILE);
        skill.setTags(mergeTags(tags, frontmatter.getTags()));
        skill.setEnabled(Boolean.TRUE);
        skill.setPackageSize((long) zipBytes.length);
        skill.setFileCount(entries.size());
        skill.setOssFilename(originalFilename);

        // 先落库拿到 id，再释放到以 id 命名的工作区目录
        aigcSkillService.save(skill);

        try {
            OssObject ossObject = ossService.upload(new OssUploadRequest(
                    originalFilename, "application/zip", zipBytes, Map.of("skillId", skill.getId())));
            skill.setOssObjectKey(ossObject.objectKey());
            skill.setLocalPath(WORKSPACE_PREFIX + skill.getId());
            materializeWorkspace(skill.getLocalPath(), entries, base);
            aigcSkillService.updateById(skill);
        } catch (RuntimeException exception) {
            cleanupQuietly(skill);
            aigcSkillService.removeById(skill.getId());
            throw exception;
        } finally {
            log.info("技能包安装完成，skillId={}, name={}, files={}", skill.getId(), name, entries.size());
        }
        return skill;
    }

    @Override
    public AigcSkill installFolder(Map<String, byte[]> files, String rootName, String tags) {
        if (files == null || files.isEmpty()) {
            throw new BizException("SKILL_PACKAGE_EMPTY", "文件夹内容为空");
        }
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (ZipOutputStream zipStream = new ZipOutputStream(buffer)) {
            for (Map.Entry<String, byte[]> entry : files.entrySet()) {
                String path = sanitizeEntryPath(entry.getKey());
                if (path.isEmpty()) {
                    continue;
                }
                zipStream.putNextEntry(new ZipEntry(path));
                zipStream.write(entry.getValue());
                zipStream.closeEntry();
            }
        } catch (IOException exception) {
            throw new BizException("SKILL_PACKAGE_ZIP_FAILED", "文件夹打包失败：" + exception.getMessage());
        }
        String packageName = StringUtils.hasText(rootName) ? rootName.trim() + ".zip" : "skill-folder.zip";
        return installPackage(buffer.toByteArray(), packageName, tags);
    }

    @Override
    public List<SkillFileNode> listFiles(AigcSkill skill) {
        Path dir = requireSkillDir(skill);
        try (Stream<Path> stream = Files.walk(dir)) {
            List<Path> paths = stream.filter(path -> !path.equals(dir)).sorted().toList();
            return buildTree(dir, paths);
        } catch (IOException exception) {
            throw new BizException("SKILL_WORKSPACE_ERROR", "读取技能工作区失败：" + exception.getMessage());
        }
    }

    @Override
    public SkillFileContent readFile(AigcSkill skill, String path) {
        Path target = resolveSecurePath(skill, path);
        try {
            if (!Files.exists(target)) {
                throw new BizException("SKILL_FILE_NOT_FOUND", "技能文件不存在：" + path);
            }
            long size = Files.size(target);
            if (Files.isDirectory(target)) {
                return new SkillFileContent(path, null, true, size);
            }
            if (isBinaryFile(target)) {
                return new SkillFileContent(path, null, true, size);
            }
            return new SkillFileContent(path, Files.readString(target, StandardCharsets.UTF_8), false, size);
        } catch (IOException exception) {
            throw new BizException("SKILL_WORKSPACE_ERROR", "读取技能文件失败：" + exception.getMessage());
        }
    }

    @Override
    public AigcSkill saveFile(AigcSkill skill, String path, String content) {
        Path target = resolveSecurePath(skill, path);
        try {
            Files.createDirectories(target.getParent());
            Files.writeString(target, content == null ? "" : content, StandardCharsets.UTF_8);
        } catch (IOException exception) {
            throw new BizException("SKILL_WORKSPACE_ERROR", "保存技能文件失败：" + exception.getMessage());
        }
        if (ENTRY_FILE.equalsIgnoreCase(path)) {
            syncFrontmatter(skill, content);
        }
        return skill;
    }

    /**
     * 读取技能包内的文档内容：
     * path 为空读入口主文档（默认 SKILL.md），传目录返回文件清单，传文件返回正文；
     * 主文档会剥离 frontmatter，非主文档原样返回，二进制文件返回占位提示。
     *
     * @param skill 技能实体
     * @param path  技能包内相对路径，null/空表示读主文档
     * @return 文档文本内容，或目录清单/二进制占位提示
     */
    @Override
    public String readSkillDocument(AigcSkill skill, String path) {
        // 确保本地工作区就绪，缺失时自动从 OSS 回源释放
        ensureWorkspace(skill);
        // 入口文件名取技能配置，缺省为 SKILL.md
        String entryFile = StringUtils.hasText(skill.getEntryFile()) ? skill.getEntryFile() : ENTRY_FILE;
        // path 为空时回退入口主文档，否则按调用方指定的相对路径读取
        String relativePath = StringUtils.hasText(path) ? path.trim() : entryFile;
        // 解析为工作区内绝对路径并做防穿越校验（阻止 .. 逃逸）
        Path target = resolveSecurePath(skill, relativePath);
        try {
            // 目标是目录：列出其中内容（子目录加 / 后缀），拼成清单文本返回
            if (Files.isDirectory(target)) {
                List<String> children = new ArrayList<>();
                try (Stream<Path> stream = Files.list(target)) {
                    stream.sorted().forEach(child -> children
                            .add(child.getFileName().toString() + (Files.isDirectory(child) ? "/" : "")));
                }
                return "目录 " + relativePath + " 下的文件：\n" + String.join("\n", children);
            }
            // 文件不存在则报错，提示具体路径便于排查
            if (!Files.exists(target)) {
                throw new BizException("SKILL_FILE_NOT_FOUND", "技能文件不存在：" + relativePath);
            }
            // 二进制文件不返回正文，避免模型收到乱码
            if (isBinaryFile(target)) {
                return "[二进制文件，无法以文本形式展示：" + relativePath + "]";
            }
            // 读取文本内容；主文档额外剥离 frontmatter（name/description 等元数据不进模型上下文）
            String content = Files.readString(target, StandardCharsets.UTF_8);
            if (entryFile.equalsIgnoreCase(relativePath)) {
                return SkillFrontmatter.stripFrontmatter(content);
            }
            return content;
        } catch (IOException exception) {
            // 读取过程中的 IO 异常统一包装为业务异常抛出
            throw new BizException("SKILL_WORKSPACE_ERROR", "读取技能文档失败：" + exception.getMessage());
        }
    }

    @Override
    public byte[] downloadPackage(AigcSkill skill) {
        if (!StringUtils.hasText(skill.getOssObjectKey())) {
            throw new BizException("SKILL_OSS_MISSING", "该技能没有 OSS 原始包");
        }
        return ossService.download(skill.getOssObjectKey());
    }

    @Override
    public void removePackage(AigcSkill skill) {
        deleteWorkspaceQuietly(skill.getLocalPath());
        if (StringUtils.hasText(skill.getOssObjectKey())) {
            ossService.delete(skill.getOssObjectKey());
        }
    }

    /**
     * 确保本地工作区存在，缺失时从 OSS 回源释放。
     */
    private void ensureWorkspace(AigcSkill skill) {
        // 本地路径缺失直接视为工作区不可用
        if (!StringUtils.hasText(skill.getLocalPath())) {
            throw new BizException("SKILL_WORKSPACE_MISSING", "技能缺少本地工作区路径");
        }
        // 工作区目录存在且非空即就绪，直接返回
        Path dir = skillDir(skill.getLocalPath());
        if (Files.isDirectory(dir)) {
            try (Stream<Path> stream = Files.list(dir)) {
                if (stream.findAny().isPresent()) {
                    return;
                }
            } catch (IOException exception) {
                log.warn("检查技能工作区失败，skillId={}", skill.getId(), exception);
            }
        }
        // 无本地目录且无 OSS 备份，无法恢复
        if (!StringUtils.hasText(skill.getOssObjectKey())) {
            throw new BizException("SKILL_OSS_MISSING", "本地技能包已丢失且没有 OSS 备份可回源");
        }
        // 从 OSS 下载技能包压缩包并解压为条目映射
        byte[] zipBytes = ossService.download(skill.getOssObjectKey());
        Map<String, byte[]> entries = unzip(zipBytes);
        // 定位包内技能根目录后，将内容物化到本地工作区
        String base = locateSkillRoot(entries);
        materializeWorkspace(skill.getLocalPath(), entries, base);
        log.info("本地技能包缺失，已从 OSS 回源，skillId={}", skill.getId());
    }

    private void syncFrontmatter(AigcSkill skill, String content) {
        SkillFrontmatter frontmatter = SkillFrontmatter.parse(content);
        if (!frontmatter.isPresent()) {
            return;
        }
        if (StringUtils.hasText(frontmatter.getName())) {
            skill.setName(sanitizeSkillName(frontmatter.getName(), skill.getName()));
        }
        if (StringUtils.hasText(frontmatter.getTitle())) {
            skill.setTitle(frontmatter.getTitle());
        }
        if (StringUtils.hasText(frontmatter.getDescription())) {
            skill.setDescription(frontmatter.getDescription());
        }
        if (StringUtils.hasText(frontmatter.getVersion())) {
            skill.setVersion(frontmatter.getVersion());
        }
        if (StringUtils.hasText(frontmatter.getLicense())) {
            skill.setLicense(frontmatter.getLicense());
        }
        if (StringUtils.hasText(frontmatter.getTags())) {
            skill.setTags(mergeTags(null, frontmatter.getTags()));
        }
    }

    /**
     * 将技能的本地相对路径转换为工作区下的绝对目录，并做防穿越校验
     * （解析结果必须仍在工作区根目录内）。
     */
    private Path skillDir(String localPath) {
        // 工作区根目录（配置项 workspaceDir）的绝对路径
        Path root = Paths.get(skillProperties.getWorkspaceDir()).toAbsolutePath().normalize();
        // 解析技能目录，超出工作区根目录即非法
        Path dir = root.resolve(localPath).normalize();
        if (!dir.startsWith(root)) {
            throw new BizException("SKILL_PATH_INVALID", "非法的技能工作区路径");
        }
        return dir;
    }

    private Path requireSkillDir(AigcSkill skill) {
        ensureWorkspace(skill);
        return skillDir(skill.getLocalPath());
    }

    /**
     * 将技能包内相对路径解析为工作区内绝对路径，并做防目录穿越校验：
     * path 为空或根 "/" 时直接返回技能根目录；解析结果必须仍在技能目录内，
     * 否则视为非法路径（阻止 ../ 逃逸访问工作区外文件）。
     */
    private Path resolveSecurePath(AigcSkill skill, String path) {
        // 空路径或根路径直接返回技能根目录
        if (!StringUtils.hasText(path) || "/".equals(path.trim())) {
            return skillDir(skill.getLocalPath());
        }
        // 从技能根目录解析目标并规范化，必须仍在技能目录内才合法
        Path dir = skillDir(skill.getLocalPath());
        Path target = dir.resolve(path.trim()).normalize();
        if (!target.startsWith(dir)) {
            throw new BizException("SKILL_PATH_INVALID", "非法的技能文件路径：" + path);
        }
        return target;
    }

    /**
     * 将技能包内容写入本地工作区。
     */
    private void materializeWorkspace(String localPath, Map<String, byte[]> entries, String base) {
        Path dir = skillDir(localPath);
        deleteDirectoryQuietly(dir);
        try {
            Files.createDirectories(dir);
            for (Map.Entry<String, byte[]> entry : entries.entrySet()) {
                String relativePath = entry.getKey().substring(base.length());
                if (relativePath.isBlank()) {
                    continue;
                }
                Path target = dir.resolve(relativePath).normalize();
                if (!target.startsWith(dir)) {
                    continue;
                }
                Files.createDirectories(target.getParent());
                Files.write(target, entry.getValue());
            }
        } catch (IOException exception) {
            throw new BizException("SKILL_WORKSPACE_ERROR", "释放技能包到工作区失败：" + exception.getMessage());
        }
    }

    /**
     * 校验并定位技能包根目录：根目录必须包含 SKILL.md，
     * 若 zip 内只有一个包裹目录且其内包含 SKILL.md，则以该目录为根。
     */
    private String locateSkillRoot(Map<String, byte[]> entries) {
        if (entries.containsKey(ENTRY_FILE)) {
            return "";
        }
        Map<String, List<String>> topDirs = new LinkedHashMap<>();
        for (String path : entries.keySet()) {
            int separator = path.indexOf('/');
            if (separator > 0) {
                topDirs.computeIfAbsent(path.substring(0, separator), key -> new ArrayList<>())
                        .add(path.substring(separator + 1));
            }
        }
        if (topDirs.size() == 1) {
            String only = topDirs.keySet().iterator().next();
            if (entries.containsKey(only + "/" + ENTRY_FILE)) {
                return only + "/";
            }
        }
        throw new BizException(
                "SKILL_PACKAGE_INVALID",
                "技能包格式不正确：包根目录必须包含 SKILL.md 入口文档（YAML frontmatter + Markdown 指令正文）");
    }

    private SkillFrontmatter parseEntryDocument(Map<String, byte[]> entries, String base) {
        byte[] bytes = entries.get(base + ENTRY_FILE);
        String markdown = new String(bytes, StandardCharsets.UTF_8);
        return SkillFrontmatter.parse(markdown);
    }

    /**
     * 解压 zip 到内存，过滤目录条目并防御 zip-slip。
     */
    private Map<String, byte[]> unzip(byte[] zipBytes) {
        Map<String, byte[]> entries = new TreeMap<>();
        try (ZipInputStream zipStream = new ZipInputStream(new ByteArrayInputStream(zipBytes))) {
            ZipEntry entry;
            while ((entry = zipStream.getNextEntry()) != null) {
                if (entry.isDirectory()) {
                    continue;
                }
                String path = sanitizeEntryPath(entry.getName());
                if (path.isEmpty()) {
                    continue;
                }
                entries.put(path, zipStream.readAllBytes());
                zipStream.closeEntry();
            }
        } catch (IOException exception) {
            throw new BizException("SKILL_PACKAGE_INVALID", "技能包解压失败，请确认上传的是有效的 zip 包");
        }
        if (entries.isEmpty()) {
            throw new BizException("SKILL_PACKAGE_INVALID", "技能包内容为空");
        }
        return entries;
    }

    private String sanitizeEntryPath(String path) {
        String normalized = path.replace('\\', '/').trim();
        while (normalized.startsWith("/")) {
            normalized = normalized.substring(1);
        }
        if (normalized.contains("..") || normalized.contains(":")) {
            throw new BizException("SKILL_PACKAGE_INVALID", "技能包内存在非法路径：" + path);
        }
        return normalized;
    }

    private String sanitizeSkillName(String name, String fallback) {
        if (!StringUtils.hasText(name)) {
            return fallback;
        }
        String sanitized = name.trim().toLowerCase()
                .replaceAll("[^a-z0-9\\u4e00-\\u9fa5_-]+", "-");
        while (sanitized.startsWith("-")) {
            sanitized = sanitized.substring(1);
        }
        while (sanitized.endsWith("-")) {
            sanitized = sanitized.substring(0, sanitized.length() - 1);
        }
        if (sanitized.length() > 64) {
            sanitized = sanitized.substring(0, 64);
        }
        return StringUtils.hasText(sanitized) ? sanitized : fallback;
    }

    private String mergeTags(String externalTags, String frontmatterTags) {
        List<String> merged = new ArrayList<>();
        for (String source : new String[] {externalTags, frontmatterTags}) {
            if (!StringUtils.hasText(source)) {
                continue;
            }
            for (String tag : source.split("[,，]")) {
                String trimmed = tag.trim();
                if (StringUtils.hasText(trimmed) && !merged.contains(trimmed)) {
                    merged.add(trimmed);
                }
            }
        }
        return String.join(",", merged);
    }

    /**
     * 判断文件是否为二进制：扩展名不在文本扩展名白名单中直接判定为二进制；
     * 无扩展名或扩展名在白名单中时再按内容探测（含 NUL 字符视为二进制）。
     */
    private boolean isBinaryFile(Path target) {
        // 先按扩展名快速判定，已知文本类型不进内容探测
        String filename = target.getFileName().toString().toLowerCase();
        int dot = filename.lastIndexOf('.');
        if (dot >= 0) {
            String ext = filename.substring(dot + 1);
            // 扩展名不在文本白名单，直接判定为二进制
            if (!TEXT_EXTENSIONS.contains(ext)) {
                return true;
            }
        }
        try {
            // 内容探测：读取全量字节，按 UTF-8 解码后出现 NUL 字符视为二进制
            byte[] head = Files.readAllBytes(target);
            String decoded = new String(head, StandardCharsets.UTF_8);
            return decoded.contains("\u0000");
        } catch (IOException exception) {
            // 读取失败按二进制处理，避免误返回不可读内容
            return true;
        }
    }

    private List<SkillFileNode> buildTree(Path dir, List<Path> paths) {
        return buildChildren(dir, paths.stream()
                .filter(path -> path.getParent().equals(dir))
                .toList(), paths);
    }

    private List<SkillFileNode> buildChildren(Path root, List<Path> children, List<Path> allPaths) {
        List<SkillFileNode> nodes = new ArrayList<>();
        for (Path child : children) {
            try {
                if (Files.isDirectory(child)) {
                    List<Path> grandChildren = allPaths.stream()
                            .filter(path -> path.getParent().equals(child))
                            .toList();
                    nodes.add(new SkillFileNode(
                            child.getFileName().toString(),
                            root.relativize(child).toString().replace('\\', '/'),
                            true,
                            0,
                            buildChildren(root, grandChildren, allPaths)));
                } else {
                    nodes.add(new SkillFileNode(
                            child.getFileName().toString(),
                            root.relativize(child).toString().replace('\\', '/'),
                            false,
                            Files.size(child),
                            List.of()));
                }
            } catch (IOException exception) {
                throw new UncheckedIOException(exception);
            }
        }
        nodes.sort(Comparator.comparing(SkillFileNode::directory).reversed()
                .thenComparing(SkillFileNode::name));
        return nodes;
    }

    private void cleanupQuietly(AigcSkill skill) {
        deleteWorkspaceQuietly(skill.getLocalPath());
        if (StringUtils.hasText(skill.getOssObjectKey())) {
            ossService.delete(skill.getOssObjectKey());
        }
    }

    private void deleteWorkspaceQuietly(String localPath) {
        if (!StringUtils.hasText(localPath)) {
            return;
        }
        deleteDirectoryQuietly(skillDir(localPath));
    }

    private void deleteDirectoryQuietly(Path dir) {
        if (!Files.exists(dir)) {
            return;
        }
        try (Stream<Path> stream = Files.walk(dir)) {
            stream.sorted(Comparator.reverseOrder()).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException exception) {
                    log.warn("删除文件失败：{}", path, exception);
                }
            });
        } catch (IOException exception) {
            log.warn("删除技能工作区失败：{}", dir, exception);
        }
    }

    private String firstNonBlank(String left, String right) {
        return StringUtils.hasText(left) ? left : right;
    }
}
