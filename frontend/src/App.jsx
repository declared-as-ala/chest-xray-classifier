import { useState, useRef, useEffect } from 'react'
import { predictImage } from './api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Button } from './components/ui/button'
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert'
import { Badge } from './components/ui/badge'
import { Progress } from './components/ui/progress'
import './index.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [predictionHistory, setPredictionHistory] = useState([])
  const [testImages, setTestImages] = useState([])
  const fileInputRef = useRef(null)

  // Load test images dynamically - ALL 100 IMAGES
  useEffect(() => {
    const normalImages = [
      'IM-0011-0001-0001.jpeg', 'IM-0016-0001.jpeg', 'IM-0029-0001.jpeg',
      'IM-0033-0001-0001.jpeg', 'IM-0041-0001.jpeg', 'IM-0046-0001.jpeg',
      'IM-0065-0001.jpeg', 'IM-0070-0001.jpeg', 'IM-0073-0001.jpeg',
      'IM-0079-0001.jpeg', 'IM-0099-0001.jpeg', 'IM-0102-0001.jpeg',
      'IM-0110-0001.jpeg', 'NORMAL2-IM-0013-0001.jpeg', 'NORMAL2-IM-0019-0001.jpeg',
      'NORMAL2-IM-0027-0001.jpeg', 'NORMAL2-IM-0028-0001.jpeg', 'NORMAL2-IM-0030-0001.jpeg',
      'NORMAL2-IM-0045-0001.jpeg', 'NORMAL2-IM-0051-0001.jpeg', 'NORMAL2-IM-0092-0001.jpeg',
      'NORMAL2-IM-0105-0001.jpeg', 'NORMAL2-IM-0120-0001.jpeg', 'NORMAL2-IM-0123-0001.jpeg',
      'NORMAL2-IM-0139-0001.jpeg', 'NORMAL2-IM-0173-0001-0001.jpeg', 'NORMAL2-IM-0196-0001.jpeg',
      'NORMAL2-IM-0201-0001.jpeg', 'NORMAL2-IM-0213-0001.jpeg', 'NORMAL2-IM-0219-0001.jpeg',
      'NORMAL2-IM-0241-0001.jpeg', 'NORMAL2-IM-0271-0001.jpeg', 'NORMAL2-IM-0278-0001.jpeg',
      'NORMAL2-IM-0288-0001.jpeg', 'NORMAL2-IM-0290-0001.jpeg', 'NORMAL2-IM-0297-0001.jpeg',
      'NORMAL2-IM-0303-0001.jpeg', 'NORMAL2-IM-0304-0001.jpeg', 'NORMAL2-IM-0311-0001.jpeg',
      'NORMAL2-IM-0313-0001.jpeg', 'NORMAL2-IM-0325-0001.jpeg', 'NORMAL2-IM-0326-0001.jpeg',
      'NORMAL2-IM-0337-0001.jpeg', 'NORMAL2-IM-0345-0001.jpeg', 'NORMAL2-IM-0346-0001.jpeg',
      'NORMAL2-IM-0354-0001.jpeg', 'NORMAL2-IM-0360-0001.jpeg', 'NORMAL2-IM-0364-0001.jpeg',
      'NORMAL2-IM-0376-0001.jpeg', 'NORMAL2-IM-0380-0001.jpeg',
    ]

    const pneumoniaImages = [
      'person103_bacteria_488.jpeg', 'person104_bacteria_491.jpeg', 'person109_bacteria_523.jpeg',
      'person109_bacteria_526.jpeg', 'person10_virus_35.jpeg', 'person112_bacteria_538.jpeg',
      'person118_bacteria_560.jpeg', 'person119_bacteria_566.jpeg', 'person122_bacteria_582.jpeg',
      'person123_bacteria_587.jpeg', 'person124_bacteria_589.jpeg', 'person124_bacteria_590.jpeg',
      'person127_bacteria_602.jpeg', 'person135_bacteria_646.jpeg', 'person138_bacteria_658.jpeg',
      'person147_bacteria_706.jpeg', 'person150_bacteria_717.jpeg', 'person152_bacteria_720.jpeg',
      'person157_bacteria_740.jpeg', 'person1613_virus_2799.jpeg', 'person1614_virus_2800.jpeg',
      'person161_bacteria_759.jpeg', 'person161_bacteria_762.jpeg', 'person1631_virus_2826.jpeg',
      'person1643_virus_2843.jpeg', 'person1660_virus_2869.jpeg', 'person1664_virus_2877.jpeg',
      'person1668_virus_2882.jpeg', 'person1_virus_12.jpeg', 'person1_virus_7.jpeg',
      'person20_virus_51.jpeg', 'person45_virus_95.jpeg', 'person61_virus_118.jpeg',
      'person63_virus_121.jpeg', 'person65_virus_123.jpeg', 'person66_virus_125.jpeg',
      'person71_virus_131.jpeg', 'person72_virus_133.jpeg', 'person75_virus_136.jpeg',
      'person80_bacteria_392.jpeg', 'person81_bacteria_395.jpeg', 'person83_bacteria_409.jpeg',
      'person83_bacteria_414.jpeg', 'person85_bacteria_423.jpeg', 'person86_bacteria_428.jpeg',
      'person86_bacteria_429.jpeg', 'person87_bacteria_434.jpeg', 'person90_bacteria_442.jpeg',
      'person94_bacteria_457.jpeg', 'person94_bacteria_458.jpeg',
    ]

    const loadedImages = [
      ...normalImages.map((img, idx) => ({
        id: `normal-${idx}`,
        name: img,
        url: `/test/NORMAL/${img}`,
        type: 'NORMAL',
        category: 'Normal X-ray'
      })),
      ...pneumoniaImages.map((img, idx) => ({
        id: `pneumonia-${idx}`,
        name: img,
        url: `/test/PNEUMONIA/${img}`,
        type: 'PNEUMONIA',
        category: 'Pneumonia X-ray'
      }))
    ]

    setTestImages(loadedImages)
  }, [])

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      processImageFile(file)
    }
  }

  const processImageFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file)
      setImagePreview(URL.createObjectURL(file))
      setPrediction(null)
      setError(null)
    } else {
      setError('Please select a valid image file')
    }
  }

  const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      processImageFile(files[0])
    }
  }

  const handleSampleImageClick = async (imageUrl, imageName) => {
    try {
      // Fetch the sample image
      const response = await fetch(imageUrl)
      if (!response.ok) {
        throw new Error('Image not found')
      }

      const blob = await response.blob()

      // Determine MIME type from blob or default to jpeg
      const mimeType = blob.type || 'image/jpeg'

      // Create a proper File object with correct MIME type
      const file = new File([blob], imageName, {
        type: mimeType,
        lastModified: Date.now()
      })

      setSelectedImage(file)
      setImagePreview(imageUrl)
      setPrediction(null)
      setError(null)
    } catch (err) {
      setError('Could not load sample image: ' + err.message)
      console.error('Error loading sample image:', err)
    }
  }

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const result = await predictImage(selectedImage)
      setPrediction(result)

      // Add to history
      const historyItem = {
        id: Date.now(),
        image: imagePreview,
        prediction: result.prediction,
        confidence: result.confidence,
        timestamp: new Date().toLocaleTimeString()
      }
      setPredictionHistory(prev => [historyItem, ...prev].slice(0, 10)) // Keep last 10
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setPrediction(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const clearHistory = () => {
    setPredictionHistory([])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-4">
            🏥 AI X-Ray Classifier
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Advanced chest X-ray analysis powered by ResNet-18 deep learning
          </p>
          <div className="mt-2 text-sm text-gray-500">
            {testImages.length > 0 && (
              <span>📊 Loaded {testImages.length} test images from dataset</span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Upload & Prediction Area */}
          <div className="lg:col-span-2">
            <Card className="shadow-2xl border-2 hover:shadow-purple-200 transition-shadow duration-300">
              <CardHeader className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-gray-800 dark:to-gray-700">
                <CardTitle className="text-2xl">Upload or Drag & Drop X-Ray</CardTitle>
                <CardDescription className="text-base">
                  Upload an image, drag and drop, or select from test images below
                </CardDescription>
              </CardHeader>
              <CardContent className="p-8">
                <div className="space-y-6">
                  {/* Drag & Drop Zone */}
                  <div
                    onDragEnter={handleDragEnter}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`border-4 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${isDragging
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20 scale-105'
                      : 'border-gray-300 hover:border-purple-400 dark:border-gray-600'
                      }`}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleImageSelect}
                      className="hidden"
                      id="file-upload"
                    />

                    <div className="space-y-4">
                      <svg
                        className="mx-auto h-16 w-16 text-gray-400"
                        stroke="currentColor"
                        fill="none"
                        viewBox="0 0 48 48"
                      >
                        <path
                          d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                          strokeWidth={2}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>

                      <div>
                        <label htmlFor="file-upload">
                          <Button
                            variant="outline"
                            size="lg"
                            className="cursor-pointer hover:bg-purple-50 border-2 border-purple-300 hover:border-purple-500 transition-all duration-300"
                            onClick={() => fileInputRef.current?.click()}
                            type="button"
                          >
                            📁 Choose File
                          </Button>
                        </label>
                        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                          or drag and drop your X-ray image here
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          PNG, JPG, JPEG up to 10MB
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Image Preview */}
                  {imagePreview && (
                    <div className="space-y-4 animate-in fade-in duration-300">
                      <div className="rounded-xl overflow-hidden border-4 border-purple-200 shadow-lg bg-gray-50 dark:bg-gray-800">
                        <img
                          src={imagePreview}
                          alt="Selected X-ray"
                          className="w-full h-auto max-h-96 object-contain"
                        />
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 text-center font-medium">
                        📄 {selectedImage?.name || 'Test image'}
                      </p>
                    </div>
                  )}

                  {/* Action Buttons */}
                  {selectedImage && (
                    <div className="flex gap-4 justify-center flex-wrap">
                      <Button
                        onClick={handlePredict}
                        disabled={loading}
                        size="lg"
                        className="min-w-[160px] bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-xl"
                      >
                        {loading ? (
                          <>
                            <svg
                              className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                              xmlns="http://www.w3.org/2000/svg"
                              fill="none"
                              viewBox="0 0 24 24"
                            >
                              <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                              ></circle>
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                              ></path>
                            </svg>
                            Analyzing...
                          </>
                        ) : (
                          <>
                            🔍 Analyze X-Ray
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={handleReset}
                        variant="outline"
                        size="lg"
                        className="min-w-[160px] hover:bg-gray-50"
                        disabled={loading}
                      >
                        🔄 Reset
                      </Button>
                    </div>
                  )}

                  {/* Error Alert */}
                  {error && (
                    <Alert variant="destructive" className="animate-in slide-in-from-top duration-300">
                      <AlertTitle className="font-semibold text-lg">⚠️ Error</AlertTitle>
                      <AlertDescription className="text-base">{error}</AlertDescription>
                    </Alert>
                  )}

                  {/* Prediction Results */}
                  {prediction && (
                    <div className="space-y-6 p-6 bg-gradient-to-br from-purple-50 to-blue-50 dark:from-gray-800 dark:to-gray-700 rounded-xl border-2 border-purple-200 animate-in slide-in-from-bottom duration-500">
                      <h3 className="text-2xl font-bold text-center text-purple-900 dark:text-purple-100">
                        🏆 Prediction Results
                      </h3>

                      <div className="space-y-4">
                        {/* Prediction Label */}
                        <div className="flex items-center justify-between p-5 bg-white dark:bg-gray-900 rounded-lg shadow-md">
                          <span className="text-lg font-medium text-gray-700 dark:text-gray-300">
                            Diagnosis:
                          </span>
                          <Badge
                            variant={prediction.prediction === 'NORMAL' ? 'default' : 'destructive'}
                            className="text-lg px-6 py-2"
                          >
                            {prediction.prediction === 'NORMAL' ? '✅ NORMAL' : '⚠️ PNEUMONIA'}
                          </Badge>
                        </div>

                        {/* Confidence Score */}
                        <div className="p-5 bg-white dark:bg-gray-900 rounded-lg shadow-md space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-lg font-medium text-gray-700 dark:text-gray-300">
                              Confidence:
                            </span>
                            <span className="text-2xl font-bold text-purple-600">
                              {(prediction.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <Progress value={prediction.confidence * 100} className="h-4" />
                        </div>

                        {/* Medical Disclaimer */}
                        <Alert className="bg-blue-50 border-blue-200 dark:bg-blue-900/20">
                          <AlertDescription className="text-sm text-blue-900 dark:text-blue-100">
                            <strong>⚕️ Medical Disclaimer:</strong> This AI prediction is for educational and research purposes only. It should not replace professional medical diagnosis. Always consult with a qualified healthcare provider for proper medical advice.
                          </AlertDescription>
                        </Alert>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Real Test Images Gallery */}
            <Card className="mt-6 shadow-xl">
              <CardHeader className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-800 dark:to-gray-700">
                <CardTitle className="text-xl">🩻 Real Test Images from Dataset</CardTitle>
                <CardDescription>
                  Click on any image to test the model ({testImages.length} images loaded)
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                {testImages.length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {testImages.map((img) => (
                      <div
                        key={img.id}
                        onClick={() => handleSampleImageClick(img.url, img.name)}
                        className="cursor-pointer group relative rounded-lg overflow-hidden border-2 border-gray-200 hover:border-purple-500 transition-all duration-300 hover:scale-105 hover:shadow-xl"
                      >
                        <div className="aspect-square bg-gray-100 dark:bg-gray-800 flex items-center justify-center overflow-hidden">
                          <img
                            src={img.url}
                            alt={img.name}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              // Fallback if image doesn't load
                              e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Ctext x="50%25" y="50%25" font-size="40" text-anchor="middle" dy=".3em"%3E🩻%3C/text%3E%3C/svg%3E'
                            }}
                          />
                        </div>
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2">
                          <p className="text-xs font-medium text-white truncate">
                            {img.name}
                          </p>
                          <Badge
                            variant={img.type === 'NORMAL' ? 'default' : 'destructive'}
                            className="mt-1 text-xs"
                          >
                            {img.type}
                          </Badge>
                        </div>
                        <div className="absolute inset-0 bg-purple-600 bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300 flex items-center justify-center">
                          <span className="text-white font-bold opacity-0 group-hover:opacity-100 transition-opacity">
                            Click to Test
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Alert className="bg-yellow-50 border-yellow-200">
                    <AlertDescription className="text-sm text-yellow-900">
                      💡 <strong>Note:</strong> No test images found. Make sure images exist in:
                      <code className="bg-yellow-100 px-2 py-1 rounded block mt-2">
                        frontend/public/test/NORMAL/<br />
                        frontend/public/test/PNEUMONIA/
                      </code>
                      Update the image filenames in App.jsx (lines 18-31) to match your actual files.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Sidebar - Prediction History */}
          <div className="lg:col-span-1">
            <Card className="shadow-xl sticky top-4">
              <CardHeader className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-gray-800 dark:to-gray-700">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xl">📊 History</CardTitle>
                  {predictionHistory.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={clearHistory}
                      className="text-xs hover:bg-red-50 hover:text-red-600"
                    >
                      Clear
                    </Button>
                  )}
                </div>
                <CardDescription>Recent predictions (last 10)</CardDescription>
              </CardHeader>
              <CardContent className="p-4 max-h-[600px] overflow-y-auto">
                {predictionHistory.length === 0 ? (
                  <div className="text-center py-8 text-gray-400">
                    <p className="text-sm">No predictions yet</p>
                    <p className="text-xs mt-2">Upload an X-ray to get started</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {predictionHistory.map((item) => (
                      <div
                        key={item.id}
                        className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow"
                      >
                        <div className="flex items-start gap-3">
                          <img
                            src={item.image}
                            alt="Prediction"
                            className="w-16 h-16 object-cover rounded border-2 border-gray-300"
                          />
                          <div className="flex-1 min-w-0">
                            <Badge
                              variant={item.prediction === 'NORMAL' ? 'default' : 'destructive'}
                              className="text-xs mb-1"
                            >
                              {item.prediction}
                            </Badge>
                            <p className="text-sm font-bold text-gray-700 dark:text-gray-300">
                              {(item.confidence * 100).toFixed(1)}% confidence
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                              🕐 {item.timestamp}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Model Info Card */}
            <Card className="mt-4 shadow-lg">
              <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-gray-800 dark:to-gray-700">
                <CardTitle className="text-lg">🧠 Model Info</CardTitle>
              </CardHeader>
              <CardContent className="p-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Architecture:</span>
                  <span className="font-medium">ResNet-18</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Framework:</span>
                  <span className="font-medium">PyTorch</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Backend:</span>
                  <span className="font-medium">FastAPI</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Classes:</span>
                  <span className="font-medium">2 (Normal/Pneumonia)</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500 dark:text-gray-400 space-y-1">
          <p>⚡ Powered by ResNet-18 Transfer Learning with ImageNet Pre-training</p>
          <p>🔧 FastAPI Backend • ⚛️ React Frontend • 🎨 shadcn/ui Components</p>
          <p className="text-xs mt-2">Built with ❤️ for Medical AI Research</p>
        </div>
      </div>
    </div>
  )
}

export default App
