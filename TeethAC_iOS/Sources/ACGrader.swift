import Foundation

/// Satu anchor referensi + embedding-nya.
struct Anchor {
    let grade: Int          // 1..10
    let assetName: String   // ac_grade_01 ...
    let embedding: [Float]
}

/// Hasil kecocokan ke satu anchor.
struct AnchorMatch: Identifiable {
    let id = UUID()
    let grade: Int
    let assetName: String
    let similarity: Float
}

/// Hasil prediksi AC untuk satu foto.
struct ACPrediction {
    let hardGrade: Int        // 1-NN (grade anchor paling mirip)
    let softGrade: Float      // grade tertimbang (ordinal)
    let matches: [AnchorMatch] // semua anchor, terurut mirip -> tidak
}

/// Port dari `predict_ac` di notebook.
enum ACGrader {

    static let temperature: Float = 0.1   // = CONFIG["temperature"]

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }   // embedding sudah L2-normalized
        return dot
    }

    static func predict(query: [Float], anchors: [Anchor]) -> ACPrediction {
        let sims = anchors.map { cosine(query, $0.embedding) }

        // hard = 1-NN
        var best = 0
        for i in 1..<sims.count where sims[i] > sims[best] { best = i }
        let hard = anchors[best].grade

        // soft = softmax(sims / T) . grades
        let maxS = sims.max() ?? 0
        var w = sims.map { expf(($0 - maxS) / temperature) }   // -maxS utk stabilitas numerik
        let sum = w.reduce(0, +)
        if sum > 0 { for i in 0..<w.count { w[i] /= sum } }
        var soft: Float = 0
        for i in 0..<anchors.count { soft += w[i] * Float(anchors[i].grade) }

        let matches = zip(anchors, sims)
            .map { AnchorMatch(grade: $0.grade, assetName: $0.assetName, similarity: $1) }
            .sorted { $0.similarity > $1.similarity }

        return ACPrediction(hardGrade: hard, softGrade: soft, matches: matches)
    }
}
