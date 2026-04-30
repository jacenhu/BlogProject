import { readdir, stat, rename } from 'fs/promises'
import { join, extname } from 'path'
import sharp from 'sharp'

const IMG_DIR = 'docs/.vitepress/dist/img'

async function* walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) {
      yield* walk(path)
    } else {
      yield path
    }
  }
}

async function optimize() {
  let saved = 0
  let count = 0

  for await (const path of walk(IMG_DIR)) {
    const ext = extname(path).toLowerCase()
    if (!['.png', '.jpg', '.jpeg'].includes(ext)) continue

    const before = (await stat(path)).size
    const tmpPath = path + '.tmp'
    const sharpInstance = sharp(path)

    try {
      if (ext === '.png') {
        await sharpInstance
          .png({ quality: 85, compressionLevel: 9, adaptiveFiltering: true })
          .toFile(tmpPath)
      } else {
        await sharpInstance
          .jpeg({ quality: 82, progressive: true, mozjpeg: true })
          .toFile(tmpPath)
      }

      await rename(tmpPath, path)
      const after = (await stat(path)).size
      const delta = before - after
      saved += delta
      count++
      console.log(`  ${path}  ${(before / 1024).toFixed(1)}KB → ${(after / 1024).toFixed(1)}KB  ${delta > 0 ? '-' : '+'}${Math.abs(delta / 1024).toFixed(1)}KB`)
    } catch (err) {
      console.error(`  ✗ ${path}:`, err.message)
    }
  }

  if (count > 0) {
    console.log(`\n✓ Optimized ${count} images, saved ${(saved / 1024 / 1024).toFixed(2)}MB`)
  } else {
    console.log('No images to optimize')
  }
}

optimize().catch(console.error)
