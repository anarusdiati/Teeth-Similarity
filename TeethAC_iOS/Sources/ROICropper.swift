import UIKit

/// Port dari `teeth_roi_box` di notebook: cari region gigi (whitish = saturasi
/// rendah + value tinggi) lalu kembalikan UIImage hasil crop.
///
/// Ini langkah paling berdampak pada kualitas (menyamakan framing anchor & query,
/// membuang bibir/gusi/kulit). Parameter samakan dengan CONFIG notebook.
enum ROICropper {

    // CONFIG (samakan dengan notebook)
    static let sThr: CGFloat = 0.30   // ambang saturasi
    static let vThr: CGFloat = 0.55   // ambang brightness
    static let dens: CGFloat = 0.12   // ambang densitas kolom/baris
    static let pad:  CGFloat = 0.10   // padding box
    static let workSize = 1024        // resolusi kerja deteksi box (parity ~1:1 dgn notebook)

    /// Crop ROI gigi. Bila deteksi gagal -> center crop (0.8 x 0.6), persis fallback notebook.
    static func crop(_ image: UIImage) -> UIImage {
        guard let (pixels, w, h) = downsampledRGBA(image, maxSide: workSize) else { return image }

        var col = [CGFloat](repeating: 0, count: w)  // rata-rata mask per kolom
        var row = [CGFloat](repeating: 0, count: h)  // rata-rata mask per baris

        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                let r = CGFloat(pixels[i]) / 255.0
                let g = CGFloat(pixels[i + 1]) / 255.0
                let b = CGFloat(pixels[i + 2]) / 255.0
                let mx = max(r, max(g, b))
                let mn = min(r, min(g, b))
                let v = mx
                let s = mx > 1e-6 ? (mx - mn) / (mx + 1e-6) : 0
                if s < sThr && v > vThr {           // piksel "gigi"
                    col[x] += 1
                    row[y] += 1
                }
            }
        }
        for x in 0..<w { col[x] /= CGFloat(h) }
        for y in 0..<h { row[y] /= CGFloat(w) }

        let colMax = col.max() ?? 0
        let rowMax = row.max() ?? 0
        let W = CGFloat(image.size.width), H = CGFloat(image.size.height)

        func centerCrop() -> UIImage {
            let cw = W * 0.8, ch = H * 0.6
            return crop(image, rect: CGRect(x: (W - cw) / 2, y: (H - ch) / 2, width: cw, height: ch))
        }

        if colMax < 1e-3 || rowMax < 1e-3 { return centerCrop() }

        // indeks kolom/baris di atas ambang densitas
        let cThr = dens * colMax, rThr = dens * rowMax
        let cx = (0..<w).filter { col[$0] > cThr }
        let ry = (0..<h).filter { row[$0] > rThr }
        guard let x0i = cx.first, let x1i = cx.last,
              let y0i = ry.first, let y1i = ry.last else { return centerCrop() }

        // skala balik dari grid kerja -> koordinat gambar asli
        let sx = W / CGFloat(w), sy = H / CGFloat(h)
        var x0 = CGFloat(x0i) * sx, x1 = CGFloat(x1i) * sx
        var y0 = CGFloat(y0i) * sy, y1 = CGFloat(y1i) * sy
        let bw = x1 - x0, bh = y1 - y0

        if bw < 0.25 * W || bh < 0.15 * H { return centerCrop() }

        let pw = bw * pad, ph = bh * pad
        x0 = max(0, x0 - pw); y0 = max(0, y0 - ph)
        x1 = min(W, x1 + pw); y1 = min(H, y1 + ph)
        return crop(image, rect: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
    }

    // MARK: helpers

    private static func crop(_ image: UIImage, rect: CGRect) -> UIImage {
        guard let cg = image.cgImage else { return image }
        let scale = CGFloat(cg.width) / image.size.width   // px per point
        let px = CGRect(x: rect.origin.x * scale, y: rect.origin.y * scale,
                        width: rect.width * scale, height: rect.height * scale)
        guard let cropped = cg.cropping(to: px) else { return image }
        return UIImage(cgImage: cropped, scale: image.scale, orientation: image.imageOrientation)
    }

    /// Render gambar ke buffer RGBA kecil (sisi terpanjang = maxSide) untuk deteksi.
    private static func downsampledRGBA(_ image: UIImage, maxSide: Int) -> ([UInt8], Int, Int)? {
        guard let cg = image.cgImage else { return nil }
        let ow = cg.width, oh = cg.height
        let scale = CGFloat(maxSide) / CGFloat(max(ow, oh))
        let w = max(1, Int((CGFloat(ow) * scale).rounded()))
        let h = max(1, Int((CGFloat(oh) * scale).rounded()))

        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return (buf, w, h)
    }
}
