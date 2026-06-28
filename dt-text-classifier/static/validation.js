/**
 * 与 src/system/validation.py 保持一致的客户端校验。
 * 常量由页面模板注入到 window.DETECT_VALIDATION。
 */
(function (global) {
  const DEFAULTS = {
    minTextLen: 6,
    maxTextLen: 10000,
    minTokenCount: 1,
    maxSpecialRatio: 0.85,
  };

  const PUNCTUATIONS = new Set(
    "，。！？；：、“”‘’（）()《》<>【】[],.!?;:'\"-".split("")
  );
  const TOKEN_RE = /[\u4e00-\u9fffA-Za-z0-9]+/g;
  const CONTROL_RE = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]/;

  function cfg() {
    return { ...DEFAULTS, ...(global.DETECT_VALIDATION || {}) };
  }

  function cleanText(text) {
    return text.trim().replace(/\u3000/g, " ").replace(/\s+/g, " ");
  }

  function countSpecial(text) {
    let n = 0;
    for (const c of text) {
      if (!c.match(/[\u4e00-\u9fffA-Za-z0-9]/) && !PUNCTUATIONS.has(c) && !/\s/.test(c)) {
        n += 1;
      }
    }
    return n;
  }

  function validateDetectText(raw) {
    const { minTextLen, maxTextLen, minTokenCount, maxSpecialRatio } = cfg();

    if (raw == null) {
      return { ok: false, text: "", error: "请输入待检测文本。" };
    }
    if (typeof raw !== "string") {
      return { ok: false, text: "", error: "text 字段必须是字符串。" };
    }
    if (raw.includes("\x00")) {
      return { ok: false, text: "", error: "文本包含非法空字符。" };
    }
    if (CONTROL_RE.test(raw)) {
      return { ok: false, text: "", error: "文本包含非法控制字符。" };
    }

    const text = cleanText(raw);
    if (!text) {
      return { ok: false, text: "", error: "请输入待检测文本。" };
    }

    const length = text.length;
    if (length < minTextLen) {
      return {
        ok: false,
        text: "",
        error: `文本过短，至少需要 ${minTextLen} 个字符（当前 ${length} 个）。`,
      };
    }
    if (length > maxTextLen) {
      return {
        ok: false,
        text: "",
        error: `文本过长，最多 ${maxTextLen} 个字符（当前 ${length} 个）。`,
      };
    }

    const tokens = text.match(TOKEN_RE) || [];
    if (tokens.length < minTokenCount) {
      return {
        ok: false,
        text: "",
        error: `有效内容不足，至少需包含 ${minTokenCount} 个中文/英文/数字词。`,
      };
    }

    if (countSpecial(text) / length > maxSpecialRatio) {
      return { ok: false, text: "", error: "特殊字符占比过高，请输入正常文本。" };
    }

    return { ok: true, text, error: "" };
  }

  global.validateDetectText = validateDetectText;
})(window);
