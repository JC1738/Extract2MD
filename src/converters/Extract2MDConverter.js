import * as pdfjsLib from 'pdfjs-dist/build/pdf.mjs';
import Tesseract from 'tesseract.js';
import WebLLMEngine from '../engines/WebLLMEngine.js';
import OutputParser from '../utils/OutputParser.js';
import SystemPrompts from '../utils/SystemPrompts.js';
import ConfigValidator from '../utils/ConfigValidator.js';
import TextProcessor from '../utils/TextProcessor.js';

// Import tiktoken for accurate token-based splitting

export class Extract2MDConverter {
    constructor(config = {}) {
        // Get default configuration and validate/normalize user-provided config
        const defaultConfig = ConfigValidator.getDefaultConfig();
        this.config = ConfigValidator.validate(config);

        // Initialize components
        this.webllmEngine = null;
        this.outputParser = new OutputParser();
        this.textProcessor = new TextProcessor();

        // Setup PDF.js worker using centralized configuration
        pdfjsLib.GlobalWorkerOptions.workerSrc = this.config.pdfJsWorkerSrc || defaultConfig.pdfJsWorkerSrc;
    }

    static async quickConvertOnly(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._performQuickConvert(pdfFile);
    }

    static async highAccuracyConvertOnly(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._performHighAccuracyConvert(pdfFile);
    }

    static async quickConvertWithLLM(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._processWithLLM(pdfFile);
    }

    static async highAccuracyConvertWithLLM(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._processWithLLM(pdfFile);
    }

    static async combinedConvertWithLLM(pdfFile, config) {
        const converter = new Extract2MDConverter(config);
        return await converter._processWithLLM(pdfFile);
    }

    async _performQuickConvert(pdfFile) {
        if (!(pdfFile instanceof File)) throw new Error('Invalid input: pdfFile must be a File object.');

        this.config.progressCallback({ stage: 'start_quick', message: 'Starting quick conversion...' });

        const rawText = await this._extractText(pdfFile);
        let cleanedText = this.textProcessor.postProcessText(rawText, this.config.postProcessRules);
        cleanedText = cleanedText.replace(/\r\n/g, '\n').replace(/\n{2,}/g, '\n\n').trim();

        this.config.progressCallback({ stage: 'markdown_quick', message: 'Converting to Markdown...' });
        const markdown = this.textProcessor.convertToMarkdown(cleanedText);

        this.config.progressCallback({ stage: 'complete_quick', message: 'Quick conversion complete.' });
        return markdown;
    }

    async _performHighAccuracyConvert(pdfFile) {
        if (!(pdfFile instanceof File)) throw new Error('Invalid input: pdfFile must be a File object.');

        this.config.progressCallback({ stage: 'start_ocr', message: 'Starting high-accuracy OCR conversion...' });

        const tesseractLang = this.config.tesseractLanguage || 'eng';
        const tesseractOpts = { ...this.config.tesseractOptions };

        let worker;
        try {
            this.config.progressCallback({ stage: 'ocr_worker_init', message: 'Initializing Tesseract OCR worker...' });
            worker = await Tesseract.createWorker(tesseractLang, 1, tesseractOpts);
            await worker.loadLanguage(tesseractLang);
            await worker.initialize(tesseractLang);

            this.config.progressCallback({ stage: 'ocr_worker_ready', message: 'OCR worker initialized successfully.' });

            const arrayBuffer = await pdfFile.arrayBuffer();
            const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            let fullTextAccumulator = '';
            const numPages = pdfDoc.numPages;

            for (let pageNum = 1; pageNum <= numPages; pageNum++) {
                this.config.progressCallback({
                    stage: 'ocr_render_page',
                    message: `Rendering page ${pageNum}/${numPages} for OCR...`,
                    currentPage: pageNum,
                    totalPages: numPages
                });

                const page = await pdfDoc.getPage(pageNum);
                const viewport = page.getViewport({ scale: 2.5 });

                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;

                await page.render({ canvasContext: context, viewport: viewport }).promise;

                this.config.progressCallback({
                    stage: 'ocr_recognize_page',
                    message: `OCR processing page ${pageNum}/${numPages}...`,
                    currentPage: pageNum,
                    totalPages: numPages
                });

                const recognition = await worker.recognize(canvas);
                const ocrPageText = recognition.data?.text || '';
                fullTextAccumulator += ocrPageText + '\n';

                canvas.width = 0;
                canvas.height = 0;
            }

            this.config.progressCallback({ stage: 'ocr_terminate_worker', message: 'Terminating Tesseract worker...' });
            await worker.terminate();

            let cleanedText = this.textProcessor.postProcessText(fullTextAccumulator, this.config.postProcessRules);
            cleanedText = cleanedText.replace(/\r\n/g, '\n').replace(/\n{2,}/g, '\n\n').trim();

            const markdown = this.textProcessor.convertToMarkdown(cleanedText);

            this.config.progressCallback({ stage: 'complete_ocr', message: 'High-accuracy conversion complete.' });
            return markdown;

        } catch (error) {
            if (worker) await worker.terminate();
            throw new Error(`OCR processing failed: ${error.message}`);
        }
    }

    async _processWithLLM(pdfFile) {
        if (!this.webllmEngine) {
            this.webllmEngine = new WebLLMEngine(this.config.llm);
            // Initialize the WebLLM engine
            await this.webllmEngine.initialize(this.config.llm.model, this.config.llm.options);
        }

        const systemPrompt = this.config.systemPrompts.combinedExtraction || '';
        const text = await this._extractText(pdfFile);

        // Check if document is too large for LLM's context window
        const maxTokens = this.config.llm.maxTokens || 4096;
        if (text.split(' ').length > maxTokens / 2) { // Heuristic: split if text is likely to exceed token limit
            this.config.progressCallback({
                stage: 'document_chunking',
                message: `Chunking document for LLM processing...`
            });

            const chunks = this._chunkDocumentForLLM(text, maxTokens);
            let fullResult = '';

            for (let i = 0; i < chunks.length; i++) {
                const chunkPrompt = `${systemPrompt}\n\n${chunks[i]}`;
                try {
                    this.config.progressCallback({
                        stage: 'llm_processing',
                        message: `Processing chunk ${i+1}/${chunks.length} with LLM...`
                    });

                    const result = await this.webllmEngine.generate(chunkPrompt, this.config.llm.options);
                    fullResult += result + '\n\n';
                } catch (error) {
                    console.error(`LLM processing failed for chunk ${i+1}:`, error);
                    throw new Error(`LLM processing failed: ${error.message}`);
                }
            }

            return fullResult.trim();
        } else {
            // For smaller documents, process as before
            const fullPrompt = `${systemPrompt}\n\n${text}`;
            try {
                const result = await this.webllmEngine.generate(fullPrompt, this.config.llm.options);
                return result;
            } catch (error) {
                console.error(`LLM processing failed:`, error);
                throw new Error(`LLM processing failed: ${error.message}`);
            }
        }
    }

    /**
     * Enhanced text extraction method that attempts multiple approaches
     * and falls back to OCR if necessary.
     */
    async _extractText(pdfFile) {
        this.config.progressCallback({ stage: 'pdf_extract', message: 'Extracting text from PDF...' });

        const arrayBuffer = await pdfFile.arrayBuffer();
        const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
        const pdf = await loadingTask.promise;

        let fullText = '';
        let pagesWithIncompleteExtraction = [];

        // First pass: Try standard text extraction
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items.map(item => item.str).join(' ');

            // Enhanced heuristic: check for common PDF extraction issues
            const isLikelyIncomplete =
                pageText.trim().length < 50 ||
                /[\u2013\u2014]/.test(pageText) || // Check for en-dash/em-dash patterns
                /\b(?:Page|Section)\s*\d+\b/i.test(pageText); // Check for pagination markers

            if (isLikelyIncomplete) {
                pagesWithIncompleteExtraction.push(i);
            }

            fullText += pageText + '\n\n'; // Add newlines between pages
            this.config.progressCallback({
                stage: 'pdf_extract',
                message: `Extracted text from page ${i}/${pdf.numPages}`,
                currentPage: i,
                totalPages: pdf.numPages
            });
        }

        // If we have pages with likely incomplete extraction, try OCR for those pages
        if (pagesWithIncompleteExtraction.length > 0) {
            this.config.progressCallback({
                stage: 'ocr_fallback',
                message: `Falling back to OCR for ${pagesWithIncompleteExtraction.length} pages...`
            });

            const tesseractLang = this.config.tesseractLanguage || 'eng';
            const tesseractOpts = { ...this.config.tesseractOptions };

            let worker;
            try {
                // Use higher quality OCR settings
                worker = await Tesseract.createWorker(tesseractLang, 1, {
                    ...tesseractOpts,
                    logger: (info) => {
                        if (info.status === 'recognizing text') {
                            this.config.progressCallback({
                                stage: 'ocr_fallback',
                                message: `OCR progress: ${Math.round(info.progress * 100)}%`
                            });
                        }
                    }
                });

                await worker.loadLanguage(tesseractLang);
                await worker.initialize(tesseractLang);

                for (let pageNum of pagesWithIncompleteExtraction) {
                    const page = await pdf.getPage(pageNum);
                    const viewport = page.getViewport({ scale: 2.5 });

                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;

                    await page.render({ canvasContext: context, viewport: viewport }).promise;

                    // Use OCR with enhanced configuration
                    const recognition = await worker.recognize(canvas, {
                        tessedit_pageseg_mode: 3, // Automatic page segmentation with OSD
                        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                    });
                    
                    const ocrPageText = recognition.data?.text || '';

                    // Replace the incomplete text with OCR result
                    const pages = fullText.split('\n\n');
                    if (pages.length > pageNum) {
                        pages[pageNum - 1] = ocrPageText;
                        fullText = pages.join('\n\n');
                    }

                    canvas.width = 0;
                    canvas.height = 0;

                    this.config.progressCallback({
                        stage: 'ocr_fallback',
                        message: `OCR fallback for page ${pageNum}/${pdf.numPages} complete`
                    });
                }

                await worker.terminate();
            } catch (error) {
                if (worker) await worker.terminate();
                console.warn('OCR fallback failed, continuing with original extraction:', error.message);
            }
        }

        this.config.progressCallback({ stage: 'pdf_extract_complete', message: 'PDF text extraction complete.' });
        return fullText;
    }

    /**
     * Split a large document into chunks that fit within the LLM's context window.
     */
    async _chunkDocumentForLLM(text, maxTokens) {
        // Use tiktoken for accurate token-based splitting
        const { encode } = await import('tiktoken');
        const encoder = await encode('cl100k_base');
        
        // Split text into tokens
        const tokens = encoder.encode(text);
        
        // Create chunks based on token limits
        const chunks = [];
        let currentChunk = [];

        for (const token of tokens) {
            currentChunk.push(token);
            
            if (currentChunk.length >= maxTokens) {
                chunks.push(encoder.decode(currentChunk));
                currentChunk = [];
            }
        }

        // Add remaining tokens as final chunk
        if (currentChunk.length > 0) {
            chunks.push(encoder.decode(currentChunk));
        }

        return chunks;
    }

}

export default Extract2MDConverter;
