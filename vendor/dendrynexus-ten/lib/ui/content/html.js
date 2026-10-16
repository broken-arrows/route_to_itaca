/* dendry
 * http://github.com/idmillington/dendry
 *
 * MIT License
 */
/*jshint indent:2 */
(function() {
  'use strict';

  var createRenderer = function(options) {
    var formatText = options && options.formatText;
    if (formatText !== undefined && typeof formatText !== 'function') {
      throw new TypeError('formatText must be a function');
    }
    var _contentObjectToHTML = function(contentObj) {
      if (contentObj.type === undefined) {
        return formatText ? formatText(contentObj) : contentObj;
      } else {
        switch (contentObj.type) {
        case 'emphasis-1':
          return '<em>' + _contentToHTML(contentObj.content) + '</em>';
        case 'emphasis-2':
          return '<strong>' + _contentToHTML(contentObj.content) + '</strong>';
        case 'emphasis-3':
          return '<code>' + _contentToHTML(contentObj.content) + '</code>';
        case 'hidden':
          return '<span class="hidden">' + _contentToHTML(contentObj.content) +
            '</span>';
        case 'line-break':
          return '<br>';

        // We can't handle elements that require state-dependency.
        // raw html for magic
        case 'magic':
          return contentObj.content;
        case 'insert':
          /* falls through */
        case 'conditional':
          throw new Error(
            contentObj.type + ' should have been evaluated by now.'
            );
        }
      }
    };

    var _contentToHTML = function(content) {
      if (Array.isArray(content)) {
        var result = [];
        for (var i = 0; i < content.length; ++i) {
          var contentObj = content[i];
          result.push(_contentObjectToHTML(contentObj));
        }
        return result.join('');
      } else {
        return _contentObjectToHTML(content);
      }
    };

    var _paragraphsToHTML = function(paragraphs) {
      var result = [];
      for (var i = 0; i < paragraphs.length; ++i) {
        var paragraph = paragraphs[i];
        switch (paragraph.type) {
        case 'heading':
          result.push('<h1>');
          result.push(_contentToHTML(paragraph.content));
          result.push('</h1>');
          break;
        case 'paragraph':
          result.push('<p>');
          result.push(_contentToHTML(paragraph.content));
          result.push('</p>');
          break;
        case 'quotation':
          result.push('<blockquote>');
          result.push(_contentToHTML(paragraph.content));
          result.push('</blockquote>');
          break;
        case 'attribution':
          result.push('<blockquote class="attribution">');
          result.push(_contentToHTML(paragraph.content));
          result.push('</blockquote>');
          break;
        case 'magic':
          result.push(paragraph.content);
          break;
        case 'hrule':
          result.push('<hr>');
          break;
        }
      }
      return result.join('');
    };

    return {
      convert: _paragraphsToHTML,
      convertLine: _contentToHTML
    };
  };

  var plain = createRenderer();
  module.exports = {
    createRenderer: createRenderer,
    convert: plain.convert,
    convertLine: plain.convertLine
  };
}());
