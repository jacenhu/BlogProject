import { readdir, stat, rename, rm } from 'node:fs/promises'
import { join, extname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import sharp from 'sharp'

const IMG_DIR = fileURLToPath(new URL('../docs/.vitepress/dist/img', import.meta.url))

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) yield* walk(path)
    else if (entry.isFile()) yield path
  }
}

export async function optimizeImages(dir = IMG_DIR) {
  let saved = 0
  let count = 0

  for await (const path of walk(dir)) {
    const ext = extname(path).toLowerCase()
    if (!['.png', '.jpg', '.jpeg'].includes(ext)) continue

    const before = (await stat(path)).size
    const tmpPath = path + '.tmp'
    try {
      const image = sharp(path)
      if (ext === '.png') {
        await image.png({ quality: 85, compressionLevel: 9, adaptiveFiltering: true }).toFile(tmpPath)
      } else {
        await image.jpeg({ quality: 82, progressive: true, mozjpeg: true }).toFile(tmpPath)
      }

      const after = (await stat(tmpPath)).size
      if (after < before) {
        await rename(tmpPath, path)
        saved += before - after
      }
      count++
    } catch (cause) {
      throw new Error(`Image optimization failed: ${path}`, { cause })
    } finally {
      await rm(tmpPath, { force: true })
    }
  }

  console.log(`Optimized ${count} images, saved ${(saved / 1024 / 1024).toFixed(2)}MB`)
  return { count, saved }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  optimizeImages(process.argv[2]).catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
}
