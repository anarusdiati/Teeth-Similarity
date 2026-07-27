import SwiftUI
import PhotosUI
import Combine

struct ContentView: View {
    @StateObject private var vm = GraderViewModel()
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    inputCard
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        Label("Pilih Foto Gigi", systemImage: "photo.on.rectangle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!vm.modelReady)

                    if !vm.modelReady && vm.errorMessage == nil {
                        ProgressView("Menyiapkan model & anchor…")
                    }
                    if vm.isProcessing { ProgressView("Menghitung grade…") }
                    if let err = vm.errorMessage {
                        Text(err).font(.footnote).foregroundStyle(.red)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    if let pred = vm.prediction { resultView(pred) }
                }
                .padding()
            }
            .navigationTitle("Teeth AC Grader")
            .task { vm.setup() }
            .onChange(of: pickerItem) { _, item in Task { await vm.loadAndGrade(item) } }
        }
    }

    private var inputCard: some View {
        Group {
            if let img = vm.inputImage {
                Image(uiImage: img).resizable().scaledToFit()
                    .frame(maxHeight: 240)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12).fill(.gray.opacity(0.15))
                    .frame(height: 200)
                    .overlay(Text("Belum ada foto").foregroundStyle(.secondary))
            }
        }
    }

    private func resultView(_ pred: ACPrediction) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading) {
                    Text("Perkiraan AC Grade").font(.caption).foregroundStyle(.secondary)
                    Text("\(pred.hardGrade)").font(.system(size: 44, weight: .bold))
                }
                Spacer()
                VStack(alignment: .trailing) {
                    Text("Tertimbang (ordinal)").font(.caption).foregroundStyle(.secondary)
                    Text(String(format: "%.1f", pred.softGrade)).font(.title2.weight(.semibold))
                }
            }
            .padding().background(.blue.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 12))

            Text("Referensi paling mirip").font(.headline)
            ForEach(Array(pred.matches.prefix(3).enumerated()), id: \.element.id) { i, m in
                HStack(spacing: 12) {
                    Image(m.assetName).resizable().scaledToFill()
                        .frame(width: 64, height: 48)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6)
                            .stroke(i == 0 ? Color.green : .clear, lineWidth: 2))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(i == 0 ? "Paling mirip" : "#\(i + 1)")  ·  Grade \(m.grade)")
                            .font(.subheadline).bold()
                        ProgressView(value: Double(max(0, m.similarity)))
                    }
                    Text(String(format: "%.3f", m.similarity))
                        .font(.caption.monospaced()).foregroundStyle(.secondary)
                }
            }

            Text("Catatan: alat bantu eksplorasi, bukan diagnosis klinis. Grading AC bersifat subjektif; target wajar adalah selisih ±1 dari klinisi.")
                .font(.caption2).foregroundStyle(.secondary).padding(.top, 4)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

@MainActor
final class GraderViewModel: ObservableObject {
    @Published var inputImage: UIImage?
    @Published var prediction: ACPrediction?
    @Published var isProcessing = false
    @Published var modelReady = false
    @Published var errorMessage: String?

    private var extractor: DINOv2FeatureExtractor?
    private var store: ReferenceStore?

    func setup() {
        guard extractor == nil else { return }
        do {
            let ex = try DINOv2FeatureExtractor()
            let st = ReferenceStore(extractor: ex)
            extractor = ex
            store = st
            st.$isReady
                .receive(on: RunLoop.main)
                .assign(to: &$modelReady)
            st.$loadError
                .receive(on: RunLoop.main)
                .sink { [weak self] in if let e = $0 { self?.errorMessage = e } }
                .store(in: &cancellables)
            st.warmUp()
        } catch {
            errorMessage = "Gagal memuat DINOv2Teeth.mlpackage. Sudah ditambahkan ke target app? (\(error.localizedDescription))"
        }
    }

    func loadAndGrade(_ item: PhotosPickerItem?) async {
        guard let item, let extractor, let store, modelReady else { return }
        errorMessage = nil
        isProcessing = true
        defer { isProcessing = false }
        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let img = UIImage(data: data) else {
                errorMessage = "Gagal memuat gambar."; return
            }
            inputImage = img
            let emb = try extractor.embedding(for: img)
            prediction = store.predict(query: emb)
        } catch {
            errorMessage = "Error: \(error.localizedDescription)"
        }
    }

    private var cancellables = Set<AnyCancellable>()
}
