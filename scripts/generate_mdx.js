const fs = require('fs');
const path = require('path');

function escapeMdx(str) {
  if (!str) return "";
  // Escape pipes to prevent breaking tables in MDX
  return str
    .replace(/\|/g, "&#124;");
}

function formatTable(parameters) {
  if (!parameters || parameters.length === 0) return "";

  let table = "| Parameter | Type | Default | Description |\n";
  table += "| :--- | :--- | :--- | :--- |\n";

  for (const p of parameters) {
    const name = p.name ? `\`${p.name}\`` : "—";
    const type = p.type ? `\`${escapeMdx(p.type)}\`` : "—";
    const defVal = p.default !== null && p.default !== undefined ? `\`${escapeMdx(p.default)}\`` : "—";
    const desc = p.description ? escapeMdx(p.description) : "—";
    table += `| ${name} | ${type} | ${defVal} | ${desc} |\n`;
  }
  return table;
}

function generateClassMdx(cls) {
  let content = `---
title: "${cls.name}"
---

# Class: ${cls.name}

${escapeMdx(cls.docstring)}

`;

  if (cls.parameters && cls.parameters.length > 0) {
    content += `## Constructor Parameters\n\n`;
    content += formatTable(cls.parameters);
    content += `\n`;
  }

  if (cls.methods && cls.methods.length > 0) {
    content += `## Methods\n\n`;
    for (const m of cls.methods) {
      content += `### ${m.name}\n\n`;
      if (m.signature) {
        content += `\`\`\`python\n${m.name}${m.signature}\n\`\`\`\n\n`;
      }
      if (m.docstring) {
        content += `${escapeMdx(m.docstring)}\n\n`;
      }
      if (m.parameters && m.parameters.length > 0) {
        content += `#### Parameters\n\n`;
        content += formatTable(m.parameters);
        content += `\n`;
      }
    }
  }
  return content;
}

function generateModuleMdx(moduleName, moduleData) {
  let content = `---
title: "${moduleName}"
---

# Module: ${moduleName}

${escapeMdx(moduleData.docstring)}

`;

  if (moduleData.classes && moduleData.classes.length > 0) {
    content += `## Classes\n\n`;
    for (const cls of moduleData.classes) {
      content += `- [${cls.name}](./${cls.name})\n`;
    }
    content += `\n`;
  }

  if (moduleData.functions && moduleData.functions.length > 0) {
    content += `## Functions\n\n`;
    for (const fn of moduleData.functions) {
      content += `### ${fn.name}\n\n`;
      if (fn.signature) {
        content += `\`\`\`python\n${fn.name}${fn.signature}\n\`\`\`\n\n`;
      }
      if (fn.docstring) {
        content += `${escapeMdx(fn.docstring)}\n\n`;
      }
      if (fn.parameters && fn.parameters.length > 0) {
        content += `#### Parameters\n\n`;
        content += formatTable(fn.parameters);
        content += `\n`;
      }
    }
  }
  return content;
}

function main() {
  const manifestPath = path.resolve(__dirname, '../docs_manifest.json');
  if (!fs.existsSync(manifestPath)) {
    console.error(`Manifest file not found at ${manifestPath}`);
    process.exit(1);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const outputBaseDir = path.resolve(__dirname, '../docs/reference');

  // Clean and recreate clintrials subdirectory under outputBaseDir
  const referenceClintrialsDir = path.join(outputBaseDir, 'clintrials');
  if (fs.existsSync(referenceClintrialsDir)) {
    fs.rmSync(referenceClintrialsDir, { recursive: true, force: true });
  }

  const modules = manifest.modules;
  for (const [moduleName, moduleData] of Object.entries(modules)) {
    const modulePath = moduleName.replace(/\./g, '/');
    const targetDir = path.join(outputBaseDir, modulePath);

    fs.mkdirSync(targetDir, { recursive: true });

    // Generate module index page
    const moduleMdxContent = generateModuleMdx(moduleName, moduleData);
    fs.writeFileSync(path.join(targetDir, 'index.mdx'), moduleMdxContent, 'utf8');

    // Generate class pages
    if (moduleData.classes) {
      for (const cls of moduleData.classes) {
        const classMdxContent = generateClassMdx(cls);
        fs.writeFileSync(path.join(targetDir, `${cls.name}.mdx`), classMdxContent, 'utf8');
      }
    }
  }

  console.log(`Successfully generated MDX documentation files under ${outputBaseDir}`);
}

main();
