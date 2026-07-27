import CoreML
import UIKit

/// Wrapper model Core ML DINOv2Teeth: UIImage (ROI) -> embedding 384-dim (L2-normalized).
final class DINOv2FeatureExtractor {

    static let inputSize = 224          // = IMG_SIZE di script konversi
    static let embedDim = 384           // = EMBED_DIM

    private let model: MLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all      // GPU / Neural Engine bila tersedia
        // DINOv2Teeth.mlpackage -> kelas Swift DINOv2Teeth (dibuat otomatis oleh Xcode)
        self.model = try DINOv2Teeth(configuration: config).model
    }

    /// ROI crop -> resize 224 -> Core ML -> embedding (L2-normalized).
    func embedding(for image: UIImage) throws -> [Float] {
        let roi = ROICropper.crop(image)
        guard let pb = roi.pixelBuffer(width: Self.inputSize, height: Self.inputSize) else {
            throw ExtractorError.pixelBufferFailed
        }
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": pb])
        let out = try model.prediction(from: input)
        guard let arr = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw ExtractorError.noOutput
        }
        var v = [Float](repeating: 0, count: arr.count)
        for i in 0..<arr.count { v[i] = arr[i].floatValue }
        // model sudah L2-normalize; re-normalize untuk aman (identik dgn notebook)
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
        if n > 0 { for i in 0..<v.count { v[i] /= n } }
        return v
    }

    enum ExtractorError: Error { case pixelBufferFailed, noOutput }
}

// MARK: - UIImage -> CVPixelBuffer (RGB, stretch resize seperti PIL .resize)

extension UIImage {
    func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return nil }
        UIGraphicsPushContext(ctx)
        draw(in: CGRect(x: 0, y: 0, width: width, height: height))   // stretch ke 224x224
        UIGraphicsPopContext()
        return buffer
    }
}
