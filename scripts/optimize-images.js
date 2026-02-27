import sharp from "sharp";
import { readdir, mkdir } from "fs/promises";
import { join } from "path";

const PUBLIC = new URL("../public", import.meta.url).pathname;
const ORIGINALS = join(PUBLIC, "originals");
const WIDTH = 1120; // 2x retina for 560px display
const QUALITY = 95;

async function main() {
  await mkdir(ORIGINALS, { recursive: true });

  const files = (await readdir(PUBLIC)).filter((f) => f.endsWith(".png"));
  console.log(`Found ${files.length} PNGs to optimize\n`);

  for (const file of files) {
    const src = join(PUBLIC, file);
    const dest = join(PUBLIC, file.replace(".png", ".webp"));
    const backup = join(ORIGINALS, file);

    const img = sharp(src);
    const meta = await img.metadata();

    // Move original to backup
    await sharp(src).toFile(backup);

    // Resize + convert to WebP
    await sharp(src)
      .resize({ width: WIDTH, withoutEnlargement: true })
      .webp({ quality: QUALITY })
      .toFile(dest);

    const { size: origSize } = await sharp(src).metadata().then(() =>
      import("fs/promises").then((fs) => fs.stat(src))
    );
    const { size: newSize } = await import("fs/promises").then((fs) =>
      fs.stat(dest)
    );

    const reduction = ((1 - newSize / origSize) * 100).toFixed(0);
    console.log(
      `${file} → ${file.replace(".png", ".webp")}  ` +
        `${(origSize / 1024).toFixed(0)}KB → ${(newSize / 1024).toFixed(0)}KB  ` +
        `(${meta.width}x${meta.height} → ${Math.min(WIDTH, meta.width)}w)  ` +
        `-${reduction}%`
    );
  }

  console.log("\nDone! Originals backed up to public/originals/");
}

main().catch(console.error);
