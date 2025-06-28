// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Psyne",
    platforms: [
        .macOS(.v11),
        .iOS(.v14),
        .tvOS(.v14),
        .watchOS(.v7)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "Psyne",
            targets: ["Psyne"]
        ),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        
        // C library wrapper target
        .target(
            name: "CPsyne",
            dependencies: [],
            path: "Sources/CPsyne",
            sources: [],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../../include"),
                .define("PSYNE_SWIFT_BINDINGS"),
            ],
            linkerSettings: [
                .linkedLibrary("psyne"),
                .linkedLibrary("pthread"),
                .linkedLibrary("m"),
                .unsafeFlags(["-L../../../build/lib"], .when(configuration: .debug)),
            ]
        ),
        
        // Swift wrapper target
        .target(
            name: "Psyne",
            dependencies: ["CPsyne"],
            path: "Sources/Psyne",
            swiftSettings: [
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ExistentialAny"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImplicitOpenExistentials"),
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        
        // Test target
        .testTarget(
            name: "PsyneTests",
            dependencies: ["Psyne"],
            path: "Tests/PsyneTests"
        ),
    ],
    cLanguageStandard: .c11,
    cxxLanguageStandard: .cxx20
)