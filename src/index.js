/**
 * Extract2MD - Enhanced PDF to Markdown conversion library
 * New API with scenario-specific methods for different use cases
 */

// Import new modular components
import Extract2MDConverter from './converters/Extract2MDConverter.js';
import WebLLMEngine from './engines/WebLLMEngine.js';
import OutputParser from './utils/OutputParser.js';
import SystemPrompts from './utils/SystemPrompts.js';
import ConfigValidator from './utils/ConfigValidator.js';

// Export new API
export default Extract2MDConverter;

// Export individual components for advanced usage
export {
    Extract2MDConverter,
    WebLLMEngine,
    OutputParser,
    SystemPrompts,
    ConfigValidator
};
