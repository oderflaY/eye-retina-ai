import SwiftUI
import AVFoundation
import Shared

@main
struct iOSApp: App {
    init() { Permisos.pedirAlAbrir() }

    var body: some Scene {
        WindowGroup {
            // La app de Compose dibuja de orilla a orilla y respeta la barra de estado y el gesto de inicio.
            ContentView().ignoresSafeArea(.all)
        }
    }
}

/** La pantalla de Compose (MainViewController.kt) dentro de SwiftUI. */
struct ContentView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        MainViewControllerKt.MainViewController()
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}
}

/** Igual que en Android: al abrir la app se pide la cámara. */
enum Permisos {
    static func pedirAlAbrir() {
        if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
            AVCaptureDevice.requestAccess(for: .video) { _ in }
        }
    }
}
