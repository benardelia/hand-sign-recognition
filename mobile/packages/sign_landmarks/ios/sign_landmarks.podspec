Pod::Spec.new do |s|
  s.name             = 'sign_landmarks'
  s.version          = '0.1.0'
  s.summary          = 'MediaPipe HolisticLandmarker -> 1662-float holistic frames.'
  s.description      = 'Runs MediaPipe Tasks HolisticLandmarker on camera / MJPEG frames for the ScribeSign app.'
  s.homepage         = 'https://github.com/benardelia/hand-sign-recognition'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'ScribeSign' => 'dev@scribesign.local' }
  s.source           = { :path => '.' }
  s.source_files     = 'sign_landmarks/Sources/sign_landmarks/**/*.{swift,h}'
  s.public_header_files = 'sign_landmarks/Sources/sign_landmarks/**/*.h'
  s.dependency 'Flutter'
  # Same MediaPipe release as Android (android/build.gradle.kts). 0.10.33's iOS xcframeworks
  # declare a .a LibraryPath but ship .framework bundles, which breaks linking.
  s.dependency 'MediaPipeTasksVision', '1.0.0'
  s.platform = :ios, '15.0'
  s.static_framework = true

  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'
  s.resource_bundles = {'sign_landmarks_privacy' => ['sign_landmarks/Sources/sign_landmarks/PrivacyInfo.xcprivacy']}
end
