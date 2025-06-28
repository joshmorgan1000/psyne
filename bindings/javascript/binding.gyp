{
  "targets": [
    {
      "target_name": "psyne_native",
      "sources": [
        "src/native/psyne_addon.cpp",
        "src/native/channel_wrapper.cpp",
        "src/native/message_wrapper.cpp",
        "src/native/metrics_wrapper.cpp",
        "src/native/compression_wrapper.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../include",
        "../../src"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "defines": [
        "NAPI_DISABLE_CPP_EXCEPTIONS"
      ],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.15"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1
        }
      },
      "conditions": [
        [
          "OS=='linux'",
          {
            "cflags_cc": [
              "-std=c++20",
              "-fPIC"
            ],
            "libraries": [
              "-L../../build",
              "-lpsyne"
            ]
          }
        ],
        [
          "OS=='mac'",
          {
            "cflags_cc": [
              "-std=c++20",
              "-stdlib=libc++"
            ],
            "libraries": [
              "-L../../build",
              "-lpsyne"
            ]
          }
        ],
        [
          "OS=='win'",
          {
            "msvs_settings": {
              "VCCLCompilerTool": {
                "AdditionalOptions": ["/std:c++20"]
              }
            },
            "libraries": [
              "../../build/psyne.lib"
            ]
          }
        ]
      ]
    }
  ]
}