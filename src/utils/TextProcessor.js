/**
 * TextProcessor.js
 * Utility for text processing and markdown conversion
 */

export class TextProcessor {
    constructor() {
        // Default post-process rules for text cleaning
        this.defaultPostProcessRules = [
            { find: /\uFB00/g, replace: 'ff' },
            { find: /\uFB01/g, replace: 'fi' },
            { find: /\uFB02/g, replace: 'fl' },
            { find: /\uFB03/g, replace: 'ffi' },
            { find: /\uFB04/g, replace: 'ffl' },
            { find: /[\u2018\u2019]/g, replace: "'" },
            { find: /[\u201C\u201D]/g, replace: '"' },
            { find: /[\u2022\u2023\u25E6\u2043\u2219\u25CF\u25CB\u2981\u2619\u2765]/g, replace: '-' },
            { find: /[\u2013\u2014]/g, replace: '-' },
            { find: /\u00AD/g, replace: '' },
            { find: /[\s\u00A0\u2000-\u200A\u202F\u205F\u3000]+/g, replace: ' ' },
        ];
    }

    /**
     * Post-process text by applying various cleanup rules
     * @param {string} text - Input text to clean
     * @param {Array} additionalRules - Additional rules to apply
     * @returns {string} Cleaned text
     */
    postProcessText(text, additionalRules = []) {
        if (!text) return '';

        let cleanedText = text;
        const allRules = [...this.defaultPostProcessRules, ...additionalRules];

        // Optimized rule application - batch similar operations
        const unicodeReplacements = [];
        const regexReplacements = [];

        for (const rule of allRules) {
            if (rule.find && typeof rule.replace === 'string') {
                if (rule.find instanceof RegExp) {
                    regexReplacements.push(rule);
                } else {
                    unicodeReplacements.push(rule);
                }
            }
        }

        // Apply unicode replacements first (typically simpler)
        for (const rule of unicodeReplacements) {
            cleanedText = cleanedText.replace(rule.find, rule.replace);
        }

        // Apply regex replacements
        for (const rule of regexReplacements) {
            cleanedText = cleanedText.replace(rule.find, rule.replace);
        }

        return cleanedText.trim();
    }

    /**
     * Convert raw text to markdown format
     * @param {string} rawText - Input text to convert
     * @returns {string} Markdown formatted text
     */
    convertToMarkdown(rawText) {
        let markdownOutputLines = [];
        const inputLines = rawText.split(/\n/);

        let currentParagraphCollector = [];
        let inPotentialTableBlock = false;
        let potentialTableBlockLines = [];

        const flushCurrentParagraph = () => {
            if (currentParagraphCollector.length > 0) {
                markdownOutputLines.push(currentParagraphCollector.join(' ').trim());
                currentParagraphCollector = [];
                this._addSeparatorLine(markdownOutputLines);
            }
        };

        const flushPotentialTableBlock = () => {
            if (potentialTableBlockLines.length > 0) {
                if (potentialTableBlockLines.length >= 2) { // Heuristic: at least 2 lines for a table/code block
                    markdownOutputLines.push('```');
                    markdownOutputLines.push(...potentialTableBlockLines.map(l => l.trimEnd()));
                    markdownOutputLines.push('```');
                } else {
                    markdownOutputLines.push(potentialTableBlockLines.join(' ').trim());
                }
                potentialTableBlockLines = [];
                this._addSeparatorLine(markdownOutputLines);
            }
            inPotentialTableBlock = false;
        };

        for (let i = 0; i < inputLines.length; i++) {
            const originalLine = inputLines[i];
            const trimmedLine = originalLine.trim();

            if (trimmedLine === '') {
                if (inPotentialTableBlock) flushPotentialTableBlock();
                flushCurrentParagraph();
                continue;
            }

            const isShortLine = trimmedLine.length > 0 && trimmedLine.length < 80;
            const noPunctuationEnd = isShortLine && !/[.,;:!?]$/.test(trimmedLine);
            const isAllCapsLine = trimmedLine.length > 2 && trimmedLine.length < 80 &&
                /^[A-Z\s\d\W]*[A-Z][A-Z\s\d\W]*$/.test(trimmedLine) && /[A-Z]/.test(trimmedLine) && !/^\d+$/.test(trimmedLine);
            const nextLineIsBlankOrEndOfFile = (i + 1 === inputLines.length || inputLines[i + 1].trim() === '');

            if (isAllCapsLine || (isShortLine && noPunctuationEnd && nextLineIsBlankOrEndOfFile && trimmedLine.length > 1)) {
                if (inPotentialTableBlock) flushPotentialTableBlock();
                flushCurrentParagraph();
                markdownOutputLines.push(`# ${trimmedLine}`);
                this._addSeparatorLine(markdownOutputLines);
                if (nextLineIsBlankOrEndOfFile && inputLines[i+1] && inputLines[i + 1].trim() === '') {
                    i++;
                }
                continue;
            }

            const hasMultipleSpacesBetweenWords = /\S\s{2,}\S/.test(originalLine);
            const hasMultipleColumnsBySpaces = originalLine.split(/\s{2,}/).length > 2 && originalLine.length > 10;

            if (hasMultipleSpacesBetweenWords || hasMultipleColumnsBySpaces) {
                flushCurrentParagraph();
                if (!inPotentialTableBlock) inPotentialTableBlock = true;
                potentialTableBlockLines.push(originalLine);
            } else {
                if (inPotentialTableBlock) flushPotentialTableBlock();
                if (trimmedLine) currentParagraphCollector.push(trimmedLine);
            }
        }

        if (inPotentialTableBlock) flushPotentialTableBlock();
        flushCurrentParagraph();

        // Optimized final cleanup - single pass to normalize excessive newlines
        return this._normalizeMarkdownNewlines(markdownOutputLines);
    }

    /**
     * Helper method to add separator lines only when needed
     * @param {Array} outputLines - Array of markdown lines
     */
    _addSeparatorLine(outputLines) {
        // Only add empty line if the last line isn't already empty
        if (outputLines.length > 0 && outputLines[outputLines.length - 1] !== '') {
            outputLines.push('');
        }
    }

    /**
     * Normalize newlines in the final markdown output
     * @param {Array} lines - Array of markdown lines
     * @returns {string} Markdown with normalized newlines
     */
    _normalizeMarkdownNewlines(lines) {
        // Filter out excessive empty lines while preserving structure
        const normalizedLines = [];
        let consecutiveEmptyLines = 0;

        for (const line of lines) {
            if (line.trim() === '') {
                consecutiveEmptyLines++;
                // Allow maximum of 1 consecutive empty line
                if (consecutiveEmptyLines <= 1) {
                    normalizedLines.push('');
                }
            } else {
                consecutiveEmptyLines = 0;
                normalizedLines.push(line.trimEnd());
            }
        }

        // Join and do final cleanup
        let finalMarkdown = normalizedLines.join('\n');
        // Remove any remaining triple+ newlines and trim
        finalMarkdown = finalMarkdown.replace(/\n{3,}/g, '\n\n').trim();
        return finalMarkdown;
    }
}

export default TextProcessor;