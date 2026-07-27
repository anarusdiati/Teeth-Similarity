import SwiftUI

/// Meng-embed 10 anchor IOTN AC (ac_grade_01..10) sekali di awal, memakai
/// code path yang IDENTIK dengan query (ROI crop + Core ML) supaya parity terjaga.
@MainActor
final class ReferenceStore: ObservableObject {

    @Published var isReady = false
    @Published var loadError: String?

    private(set) var anchors: [Anchor] = []
    private let extractor: DINOv2FeatureExtractor

    // ac_grade_01 = grade 1 ... ac_grade_10 = grade 10
    private let anchorNames: [String] = (1...10).map { String(format: "ac_grade_%02d", $0) }

    init(extractor: DINOv2FeatureExtractor) {
        self.extractor = extractor
    }

    func warmUp() {
        Task.detached(priority: .userInitiated) { [extractor, anchorNames] in
            var built: [Anchor] = []
            var missing: [String] = []
            for (idx, name) in anchorNames.enumerated() {
                guard let img = UIImage(named: name) else { missing.append(name); continue }
                if let emb = try? extractor.embedding(for: img) {
                    built.append(Anchor(grade: idx + 1, assetName: name, embedding: emb))
                }
            }
            let result = built
            let miss = missing
            await MainActor.run {
                self.anchors = result
                if !miss.isEmpty {
                    self.loadError = "Anchor belum ada di Assets: \(miss.joined(separator: ", "))"
                }
                self.isReady = result.count == 10
            }
        }
    }

    func predict(query: [Float]) -> ACPrediction {
        ACGrader.predict(query: query, anchors: anchors)
    }
}
