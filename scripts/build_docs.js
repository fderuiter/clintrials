const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

// Configure marked options
marked.setOptions({
  gfm: true,
  breaks: true,
});

const referenceDir = path.resolve(__dirname, '../docs/reference');
const distDir = path.resolve(__dirname, '../docs/dist');

// Create clean dist directory
if (fs.existsSync(distDir)) {
  fs.rmSync(distDir, { recursive: true, force: true });
}
fs.mkdirSync(distDir, { recursive: true });

const searchIndex = [];
const publicModules = [];

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

      // Wrap in HTML template
      const pageHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title} | Clintrials Documentation</title>
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
                <a href="/index.html">🏠 Home</a>
                ${breadcrumbs}
            </div>
            <div>
                <a href="/search.html">🔍 Search Reference</a>
            </div>
        </nav>
        <article>
            ${htmlContent}
        </article>
    </div>
</body>
</html>`;

      // Save HTML to dist with same relative path
      const destHtmlRelPath = itemRelPath.replace(/\.mdx$/, '.html').replace(/\\/g, '/');
      const destHtmlPath = path.join(distDir, destHtmlRelPath);
      const destHtmlDir = path.dirname(destHtmlPath);

      fs.mkdirSync(destHtmlDir, { recursive: true });
      fs.writeFileSync(destHtmlPath, pageHtml, 'utf8');

      // Keep track of module index pages for home directory links
      if (item.name === 'index.mdx' && title.startsWith('clintrials.')) {
        publicModules.push({
          name: title,
          url: '/' + destHtmlRelPath
        });
      }

      // Add clean text to search index
      const cleanContent = markdownBody
        .replace(/[#*`_\[\]()\-|]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

      searchIndex.push({
        title: title,
        url: '/' + destHtmlRelPath,
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
            <a href="/index.html">🏠 Home</a>
        </nav>
        <h1>Search Clintrials API Reference</h1>
        <input type="text" id="search-input" placeholder="Type to search (e.g., stats, ProbabilityDensitySample, crm, spending)..." autofocus>
        <div id="results"></div>
    </div>

    <script>
        let index = [];
        fetch('/search_index.json')
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

const indexHtmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clintrials API Reference Portal</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            max-width: 900px;
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
        .modules-list {
            text-align: left;
            max-width: 500px;
            margin: 0 auto;
            border-top: 1px solid #eaecef;
            padding-top: 2rem;
        }
        .modules-list h3 {
            color: #24292e;
            margin-bottom: 1rem;
        }
        ul {
            padding-left: 1.5rem;
            line-height: 1.8;
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
        <h1>Clintrials API Reference</h1>
        <p>A modernized, extremely fast, and fully searchable documentation portal generated directly from runtime docstrings.</p>
        <div>
            <a href="/search.html" class="btn">🔍 Search Reference Pages</a>
        </div>
        <div class="modules-list">
            <h3>Discovered Public Modules</h3>
            <ul>
                ${moduleLinks}
            </ul>
        </div>
    </div>
</body>
</html>`;
fs.writeFileSync(path.join(distDir, 'index.html'), indexHtmlContent, 'utf8');

console.log(`Successfully compiled MDX documentation to static HTML under ${distDir}`);
