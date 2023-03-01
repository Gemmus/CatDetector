package com.say.pytorchkotlindemo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.databinding.DataBindingUtil
import com.say.pytorchkotlindemo.databinding.ActivityMainBinding
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


const val GALLERY = 1
const val CAMERA = 2
const val TOP_K = 1
const val THRESHOLD = 9.5

typealias ClassificationListener = (classification: String) -> Unit

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Obtain binding object using the Data Binding library
        viewBinding = DataBindingUtil.setContentView<ActivityMainBinding>(
            this, R.layout.activity_main
        )

        // Set the LifecycleOwner to be able to observe LiveData objects
        viewBinding.lifecycleOwner = this

        // Bind ViewModel
        viewBinding.listeners = Listeners(this)

        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)  // This makes sure that we analyze only as fast as the analyzer can process
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, CatAnalyzer { classification ->
                        Log.d(TAG, "Classification: $classification")
                    })
                }

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }
    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    inner class CatAnalyzer(private val listener: ClassificationListener) : ImageAnalysis.Analyzer {
        /**
         * This is a custom extension to the ImageProxy class for creating bitmaps.
         */
        private var module: Module? = null
        private var inputTensor: Tensor? = null

        fun ImageProxy.toBitmap(): Bitmap {
            val yBuffer = planes[0].buffer // Y
            val vuBuffer = planes[2].buffer // VU

            val ySize = yBuffer.remaining()
            val vuSize = vuBuffer.remaining()

            val nv21 = ByteArray(ySize + vuSize)

            yBuffer.get(nv21, 0, ySize)
            vuBuffer.get(nv21, ySize, vuSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }

        override fun analyze(image: ImageProxy) {
            val bmp = image.toBitmap()
            val bmpResized = resize(bmp, 244,244)
            val classification = classify(bmpResized)

            // The shown text needs to be updated on the UI thread
            runOnUiThread(Runnable {
                viewBinding.tvResult.text = classification
            })
            listener(classification)
            image.close()
        }

        /**
         * Resize the image to fit the model input size.
         */
        private fun resize(image: Bitmap, maxWidth: Int, maxHeight: Int): Bitmap {
            var image = image
            return if (maxHeight > 0 && maxWidth > 0) {
                val width = image.width
                val height = image.height
                val ratioBitmap = width.toFloat() / height.toFloat()
                val ratioMax = maxWidth.toFloat() / maxHeight.toFloat()
                var finalWidth = maxWidth
                var finalHeight = maxHeight
                if (ratioMax > ratioBitmap) {
                    finalWidth = (maxHeight.toFloat() * ratioBitmap).toInt()
                } else {
                    finalHeight = (maxWidth.toFloat() / ratioBitmap).toInt()
                }
                image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true)
                image
            } else {
                image
            }
        }

        /**
         * This performs the actual classification with pytorch.
         */
        fun classify(bitmap: Bitmap) : String {
            module = Module.load("/data/user/0/com.say.pytorchkotlindemo/files/model_1.pt")
            val startTime = SystemClock.elapsedRealtime()
            inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )
            val moduleForwardStartTime = SystemClock.elapsedRealtime()
            val outputTensor = module?.forward(IValue.from(inputTensor))?.toTensor()
            val moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime
            println(moduleForwardDuration)

            var classification = ""
            val scores = outputTensor?.dataAsFloatArray
            scores?.let {
                val ixs = Utils.topK(scores, TOP_K)
                val topKClassNames = Array(TOP_K) { i -> (i * i).toString() }
                val topKScores = FloatArray(TOP_K)
                for (i in 0 until TOP_K) {
                    val ix = ixs[i]
                    if (ix <= Constants.IMAGE_NET_CLASSNAME.size) {
                        topKClassNames[i] = Constants.IMAGE_NET_CLASSNAME[ix]
                    }
                    topKScores[i] = scores[ix]
                }
                val analysisDuration = SystemClock.elapsedRealtime() - startTime
                println(topKScores[0])
                if (topKScores[0] >= THRESHOLD) {
                    classification = topKClassNames[0]
                }

            }
            return classification
        }
    }
}
