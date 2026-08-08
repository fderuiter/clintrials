const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

const hljs = require('highlight.js');

// Configure marked options
marked.setOptions({
  gfm: true,
  breaks: true,
});

marked.use({
  renderer: {
    code(code, infostring) {
      const lang = (infostring || '').match(/\S*/)[0];
      const normalizedLang = lang.trim().toLowerCase();

      if (normalizedLang === 'python' || normalizedLang === 'py') {
        try {
          const highlighted = hljs.highlight(code, { language: 'python' }).value;
          return `<pre><code class="language-python">${highlighted}</code></pre>`;
        } catch (err) {
          console.error('Error highlighting Python code:', err);
        }
      }

      // Fall back to plain-text formatting for unspecified or non-python languages
      const escapedCode = code
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');

      const codeClass = normalizedLang ? `class="language-${normalizedLang}"` : '';
      return `<pre><code ${codeClass}>${escapedCode}</code></pre>`;
    }
  }
});

const referenceDir = path.resolve(__dirname, '../docs/reference');
const distDir = path.resolve(__dirname, '../docs/dist');

let basePath = process.env.DOCS_BASE_PATH || process.env.BASE_PATH || '';
if (basePath) {
  if (!basePath.startsWith('/')) {
    basePath = '/' + basePath;
  }
  if (basePath.endsWith('/')) {
    basePath = basePath.slice(0, -1);
  }
}
const mdxPrefix = basePath + '/reference';

// Create clean dist directory
if (fs.existsSync(distDir)) {
  fs.rmSync(distDir, { recursive: true, force: true });
}
fs.mkdirSync(distDir, { recursive: true });

const searchIndex = [];
const publicModules = [];
const tutorials = [];

function walkAndCompile(currentDir, relativePath = "") {
  if (!fs.existsSync(currentDir)) {
    return;
  }
  const items = fs.readdirSync(currentDir, { withFileTypes: true });

  for (const item of items) {
    const itemPath = path.join(currentDir, item.name);
    const itemRelPath = relativePath ? path.join(relativePath, item.name) : item.name;

    if (item.isDirectory()) {
      walkAndCompile(itemPath, itemRelPath);
    } else if (item.name.endsWith('.mdx')) {
      const content = fs.readFileSync(itemPath, 'utf8');

      // Parse frontmatter
      let title = item.name.replace(/\.mdx$/, '');
      let markdownBody = content;

      // Extract title from frontmatter if present
      const frontmatterMatch = content.match(/^---\r?\ntitle:\s*"([^"]+)"\r?\n---\r?\n([\s\S]*)$/);
      if (frontmatterMatch) {
        title = frontmatterMatch[1];
        markdownBody = frontmatterMatch[2];
      }

      // Compile markdown to HTML
      let htmlContent = marked.parse(markdownBody);

      // Rewrite relative links to use .html extensions
      function rewriteRelativeLink(href) {
        if (!href) return href;
        if (
          href.startsWith('http://') ||
          href.startsWith('https://') ||
          href.startsWith('mailto:') ||
          href.startsWith('#') ||
          href.includes('://')
        ) {
          return href;
        }
        let [pathPart, hashPart] = href.split('#');
        let [mainPath, queryPart] = pathPart.split('?');
        if (!mainPath) return href;

        const ext = path.extname(mainPath);
        if (!ext) {
          if (mainPath.endsWith('/')) {
            mainPath = mainPath.slice(0, -1);
          }
          mainPath += '.html';
        } else if (['.md', '.mdx', '.rst'].includes(ext)) {
          mainPath = mainPath.slice(0, -ext.length) + '.html';
        }

        let newHref = mainPath;
        if (queryPart !== undefined) {
          newHref += '?' + queryPart;
        }
        if (hashPart !== undefined) {
          newHref += '#' + hashPart;
        }
        return newHref;
      }

      htmlContent = htmlContent.replace(/href="([^"]*)"/g, (match, href) => {
        return `href="${rewriteRelativeLink(href)}"`;
      }).replace(/href='([^']*)'/g, (match, href) => {
        return `href='${rewriteRelativeLink(href)}'`;
      });

      // Simple navigation breadcrumbs
      const parts = itemRelPath.replace(/\\/g, '/').replace(/\.mdx$/, '').split('/');
      const breadcrumbs = parts.map((part, index) => {
        if (index === parts.length - 1 && part === 'index') {
          return '';
        }
        return `<span style="color: #666;"> &gt; </span> ${part}`;
      }).join('');

      const destHtmlRelPath = itemRelPath.replace(/\.mdx$/, '.html').replace(/\\/g, '/');
      const depth = destHtmlRelPath.split('/').length - 1;
      const relativePrefix = depth > 0 ? '../'.repeat(depth) : './';

      // Wrap in HTML template
      const pageHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title} | Clintrials Documentation</title>
    <link rel="stylesheet" href="${relativePrefix}_static/custom.css">
    <script src="${relativePrefix}_static/custom.js" defer></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem 1rem;
            background-color: #f7f9fa;
        }
        .container {
            background: #ffffff;
            padding: 2.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid #e1e4e8;
        }
        nav {
            margin-bottom: 2rem;
            font-size: 0.95rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 1rem;
        }
        nav a {
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
        }
        nav a:hover {
            text-decoration: underline;
        }
        h1, h2, h3, h4 {
            color: #24292e;
            font-weight: 600;
            margin-top: 1.8rem;
            margin-bottom: 0.8rem;
        }
        h1 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }
        pre {
            background-color: #f6f8fa;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            border: 1px solid #e1e4e8;
        }
        code {
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.85em;
            background-color: rgba(27,31,35,0.05);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
        }
        pre code {
            background-color: transparent;
            padding: 0;
            font-size: 0.9em;
        }
        /* Static Syntax Highlighting (GitHub Light Theme) */
        .hljs-doctag,
        .hljs-keyword,
        .hljs-meta .hljs-keyword,
        .hljs-template-tag,
        .hljs-template-variable,
        .hljs-type,
        .hljs-variable.language_ {
            color: #d73a49;
        }
        .hljs-title,
        .hljs-title.class_,
        .hljs-title.class_.inherited__,
        .hljs-title.function_ {
            color: #6f42c1;
        }
        .hljs-attr,
        .hljs-attribute,
        .hljs-literal,
        .hljs-meta,
        .hljs-number,
        .hljs-operator,
        .hljs-variable,
        .hljs-selector-attr,
        .hljs-selector-class,
        .hljs-selector-id {
            color: #005cc5;
        }
        .hljs-regexp,
        .hljs-string,
        .hljs-meta .hljs-string {
            color: #032f62;
        }
        .hljs-built_in,
        .hljs-symbol {
            color: #e36209;
        }
        .hljs-comment,
        .hljs-code,
        .hljs-formula {
            color: #6a737d;
        }
        .hljs-name,
        .hljs-quote,
        .hljs-selector-tag,
        .hljs-selector-pseudo {
            color: #22863a;
        }
        .hljs-subst {
            color: #24292e;
        }
        .hljs-section {
            color: #005cc5;
            font-weight: bold;
        }
        .hljs-bullet {
            color: #735c0f;
        }
        .hljs-emphasis {
            font-style: italic;
        }
        .hljs-strong {
            font-weight: bold;
        }
        .hljs-addition {
            color: #22863a;
            background-color: #f0fff4;
        }
        .hljs-deletion {
            color: #b31d28;
            background-color: #ffeef0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
        }
        th, td {
            border: 1px solid #dfe2e5;
            padding: 10px 14px;
            text-align: left;
        }
        th {
            background-color: #f6f8fa;
        }
        tr:nth-child(even) {
            background-color: #fcfcfc;
        }
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <div>
                <a href="${mdxPrefix}/index.html">🏠 Home</a>
                ${breadcrumbs}
            </div>
            <div>
                <a href="${mdxPrefix}/search.html">🔍 Search Reference</a>
            </div>
        </nav>
        <article>
            ${htmlContent}
        </article>
    </div>
</body>
</html>`;

      // Save HTML to dist with same relative path
      const destHtmlPath = path.join(distDir, destHtmlRelPath);
      const destHtmlDir = path.dirname(destHtmlPath);

      fs.mkdirSync(destHtmlDir, { recursive: true });
      fs.writeFileSync(destHtmlPath, pageHtml, 'utf8');

      // Keep track of module index pages for home directory links
      if (item.name === 'index.mdx' && title.startsWith('clintrials.')) {
        publicModules.push({
          name: title,
          url: mdxPrefix + '/' + destHtmlRelPath
        });
      }

      if (destHtmlRelPath.startsWith('tutorials/')) {
        tutorials.push({
          title: title,
          url: mdxPrefix + '/' + destHtmlRelPath
        });
      }

      // Add clean text to search index
      const cleanContent = markdownBody
        .replace(/[#*`_\[\]()\-|]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

      searchIndex.push({
        title: title,
        url: mdxPrefix + '/' + destHtmlRelPath,
        content: cleanContent
      });
    }
  }
}

// Walk and compile reference
walkAndCompile(referenceDir);

// Write search index JSON
fs.writeFileSync(path.join(distDir, 'search_index.json'), JSON.stringify(searchIndex, null, 2), 'utf8');

// Generate search.html page
const searchHtmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Clintrials API Reference</title>
    <link rel="stylesheet" href="./_static/custom.css">
    <script src="./_static/custom.js" defer></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
            background-color: #f7f9fa;
        }
        .container {
            background: #ffffff;
            padding: 2.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid #e1e4e8;
        }
        nav {
            margin-bottom: 2rem;
            font-size: 0.95rem;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 1rem;
        }
        nav a {
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
        }
        input {
            width: 100%;
            padding: 14px;
            font-size: 1.1rem;
            border: 1px solid #d1d5da;
            border-radius: 6px;
            box-sizing: border-box;
            margin-bottom: 2rem;
            box-shadow: inset 0 1px 2px rgba(27,31,35,0.075);
        }
        input:focus {
            border-color: #0366d6;
            outline: none;
            box-shadow: 0 0 0 3px rgba(3,102,214,0.3);
        }
        .result-item {
            margin-bottom: 1.5rem;
            padding: 1rem;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            background-color: #fcfcfc;
        }
        .result-item:hover {
            border-color: #0366d6;
            background-color: #ffffff;
        }
        .result-item a {
            font-size: 1.25rem;
            color: #0366d6;
            text-decoration: none;
            font-weight: 600;
        }
        .result-item a:hover {
            text-decoration: underline;
        }
        .result-item .snippet {
            font-size: 0.95rem;
            color: #586069;
            margin-top: 0.5rem;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="${mdxPrefix}/index.html">🏠 Home</a>
        </nav>
        <h1>Search Clintrials API Reference</h1>
        <input type="text" id="search-input" placeholder="Type to search (e.g., stats, ProbabilityDensitySample, crm, spending)..." autofocus>
        <div id="results"></div>
    </div>

    <script>
        let index = [];
        fetch('${mdxPrefix}/search_index.json')
            .then(res => res.json())
            .then(data => { index = data; })
            .catch(err => console.error('Error loading search index:', err));

        const input = document.getElementById('search-input');
        const resultsDiv = document.getElementById('results');

        input.addEventListener('input', () => {
            const query = input.value.toLowerCase().trim();
            if (!query) {
                resultsDiv.innerHTML = '';
                return;
            }
            const matches = index.filter(item => 
                item.title.toLowerCase().includes(query) || 
                item.content.toLowerCase().includes(query)
            );

            if (matches.length === 0) {
                resultsDiv.innerHTML = '<p style="color: #666;">No matching pages found.</p>';
                return;
            }

            resultsDiv.innerHTML = matches.map(m => {
                // Find matching snippet
                const contentLower = m.content.toLowerCase();
                const idx = contentLower.indexOf(query);
                let snippet = m.content.slice(0, 150) + '...';
                if (idx !== -1) {
                    const start = Math.max(0, idx - 50);
                    const end = Math.min(m.content.length, idx + 100);
                    snippet = (start > 0 ? '...' : '') + m.content.slice(start, end) + '...';
                }
                return \`
                    <div class="result-item">
                        <a href="\${m.url}">\${m.title}</a>
                        <div class="snippet">\${snippet}</div>
                    </div>
                \`;
            }).join('');
        });
    </script>
</body>
</html>`;
fs.writeFileSync(path.join(distDir, 'search.html'), searchHtmlContent, 'utf8');

// Generate home index.html landing page
publicModules.sort((a, b) => a.name.localeCompare(b.name));
const moduleLinks = publicModules.map(m => `<li><a href="${m.url}">${m.name}</a></li>`).join('\n            ');

tutorials.sort((a, b) => a.title.localeCompare(b.title));
const tutorialLinks = tutorials.map(t => `<li><a href="${t.url}">${t.title}</a></li>`).join('\n            ');

const indexHtmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clintrials Documentation Portal</title>
    <link rel="stylesheet" href="./_static/custom.css">
    <script src="./_static/custom.js" defer></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 3rem 1rem;
            background-color: #f7f9fa;
        }
        .container {
            background: #ffffff;
            padding: 3rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            text-align: center;
            border: 1px solid #e1e4e8;
        }
        h1 {
            color: #24292e;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }
        p {
            color: #586069;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        a.btn {
            display: inline-block;
            background: #0366d6;
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
            box-shadow: 0 1px 0 rgba(27,31,35,0.04), inset 0 1px 0 rgba(255,255,255,0.25);
            margin-bottom: 2.5rem;
        }
        a.btn:hover {
            background: #0255b3;
        }
        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2.5rem;
            text-align: left;
            margin-top: 1rem;
            border-top: 1px solid #eaecef;
            padding-top: 2rem;
        }
        @media (max-width: 768px) {
            .grid-container {
                grid-template-columns: 1fr;
            }
        }
        .tutorials-section, .modules-list {
            background: #fafbfc;
            padding: 1.5rem;
            border-radius: 6px;
            border: 1px solid #e1e4e8;
        }
        .tutorials-section h3, .modules-list h3 {
            color: #24292e;
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.3rem;
            border-bottom: 2px solid #eaecef;
            padding-bottom: 0.5rem;
        }
        ul {
            padding-left: 1.5rem;
            line-height: 1.8;
            margin: 0;
        }
        ul li a {
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
        }
        ul li a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Clintrials Documentation Portal</h1>
        <p>A modernized, extremely fast, and fully searchable documentation portal generated directly from runtime docstrings and clinical tutorials.</p>
        <div>
            <a href="${mdxPrefix}/search.html" class="btn">🔍 Search Reference & Tutorials</a>
        </div>

        <div id="homepage-sim-placeholder" style="width: 100%; height: 800px; border: 1px solid #e1e4e8; border-radius: 8px; background: #f8f9fa; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 24px;">
            <!-- Mock window header -->
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 40px; background: #e9ecef; border-bottom: 1px solid #e1e4e8; display: flex; align-items: center; padding: 0 15px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: #ff5f56; margin-right: 8px;"></div>
                <div style="width: 12px; height: 12px; border-radius: 50%; background: #ffbd2e; margin-right: 8px;"></div>
                <div style="width: 12px; height: 12px; border-radius: 50%; background: #27c93f;"></div>
            </div>
            <!-- Mock dashboard body -->
            <div style="width: 80%; height: 60%; background: white; border: 1px dashed #ccc; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-direction: column; opacity: 0.6;">
                 <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#496D89" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                 <p style="margin-top: 15px; color: #496D89; font-size: 18px; font-family: sans-serif;">Clinical Trials Simulation Hub Dashboard</p>
            </div>
            <!-- Call to action button -->
            <button id="launch-sim-btn" style="position: absolute; z-index: 10; padding: 14px 28px; background: #496D89; color: white; border: none; border-radius: 28px; font-size: 16px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 12px rgba(73, 109, 137, 0.4); transition: transform 0.1s, background 0.2s; display: flex; align-items: center; gap: 8px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Launch Simulator
            </button>
            <!-- Loading spinner -->
            <div id="sim-loading-spinner" style="display: none; position: absolute; z-index: 5; flex-direction: column; align-items: center;">
                <div style="width: 40px; height: 40px; border: 4px solid #e1e4e8; border-top: 4px solid #496D89; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                <p style="margin-top: 12px; color: #496D89; font-weight: 600; font-family: sans-serif;">Loading Simulator...</p>
            </div>
            <style>
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                #launch-sim-btn:hover { background: #385b73; transform: scale(1.05); }
                #launch-sim-btn:active { transform: scale(0.95); }
            </style>
        </div>

        <div class="grid-container">
            <div class="tutorials-section">
                <h3>📚 Tutorials & Onboarding Guides</h3>
                <ul>
                    ${tutorialLinks}
                </ul>
            </div>
            <div class="modules-list">
                <h3>🔍 Discovered Public Modules</h3>
                <ul>
                    ${moduleLinks}
                </ul>
            </div>
        </div>
    </div>
</body>
</html>`;
fs.writeFileSync(path.join(distDir, 'index.html'), indexHtmlContent, 'utf8');

// Copy docs/_static recursively to docs/dist/_static and docs/dist/clintrials/_static
const staticSrc = path.resolve(__dirname, '../docs/_static');
const staticDestRoot = path.resolve(distDir, '_static');
const staticDestClintrials = path.resolve(distDir, 'clintrials/_static');

fs.cpSync(staticSrc, staticDestRoot, { recursive: true, force: true });
fs.cpSync(staticSrc, staticDestClintrials, { recursive: true, force: true });

// Copy docs/_extra contents directly to docs/dist/
const extraSrc = path.resolve(__dirname, '../docs/_extra');
if (fs.existsSync(extraSrc)) {
  const extraItems = fs.readdirSync(extraSrc);
  for (const item of extraItems) {
    const srcPath = path.join(extraSrc, item);
    const destPath = path.join(distDir, item);
    if (item === 'hub') {
      // If it's the hub symlink, we copy the actual /app/hub directory recursively to docs/dist/hub and docs/dist/clintrials/hub!
      const realHubSrc = path.resolve(__dirname, '../hub');
      fs.cpSync(realHubSrc, path.join(distDir, 'hub'), { recursive: true, force: true });
      fs.cpSync(realHubSrc, path.join(distDir, 'clintrials/hub'), { recursive: true, force: true });
    } else {
      fs.cpSync(srcPath, destPath, { recursive: true, force: true });
    }
  }
}

console.log(`Successfully compiled MDX documentation to static HTML under ${distDir}`);
