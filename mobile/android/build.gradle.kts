allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// sign_landmarks ships protobuf-java (MediaPipe Holistic needs the full Any API); drop the lite
// runtime everywhere so the two never end up in the APK together.
allprojects {
    configurations.all {
        exclude(group = "com.google.protobuf", module = "protobuf-javalite")
    }
}

// reactive_ble_mobile hardcodes compileSdkVersion 37, which this AGP resolves as the
// "android-37" hash while the SDK ships it as "android-37.0". Compile it against 36 like
// the rest of the app (it uses no API 37 symbols).
subprojects {
    if (name == "reactive_ble_mobile") {
        afterEvaluate {
            extensions.findByType(com.android.build.api.dsl.LibraryExtension::class.java)?.compileSdk = 36
            // Its generated lite protos compile against protobuf-java once javalite is excluded (below).
            dependencies.add("implementation", "com.google.protobuf:protobuf-java:4.26.1")
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
