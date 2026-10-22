group = "com.scribesign.sign_landmarks"
version = "1.0-SNAPSHOT"

buildscript {
    val kotlinVersion = "2.4.0"
    repositories {
        google()
        mavenCentral()
    }

    dependencies {
        classpath("com.android.tools.build:gradle:9.1.0")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlinVersion")
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

plugins {
    id("com.android.library")
}

android {
    namespace = "com.scribesign.sign_landmarks"

    compileSdk = 36

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin")
        }
    }

    defaultConfig {
        minSdk = 24
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    // Same MediaPipe release as the iOS pod (ios/sign_landmarks.podspec).
    implementation("com.google.mediapipe:tasks-vision:1.0.0") {
        exclude(group = "com.google.protobuf", module = "protobuf-javalite")
    }
    // MediaPipe's HolisticLandmarker calls Any.Builder.build() with the full-protobuf signature,
    // which protobuf-javalite lacks (NoSuchMethodError at createFromOptions). protobuf-java is a
    // superset that also contains the lite runtime. Apps must exclude protobuf-javalite globally
    // (see mobile/android/build.gradle.kts).
    api("com.google.protobuf:protobuf-java:4.26.1")
    // Classic TFLite Interpreter for the sign classifier (own .so, no clash with MediaPipe).
    implementation("com.google.ai.edge.litert:litert:1.4.2")
}
