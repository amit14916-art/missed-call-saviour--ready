package com.missedcallsaviour.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.missedcallsaviour.app.ui.theme.*
import com.missedcallsaviour.app.network.*
import com.missedcallsaviour.app.data.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        checkAndRequestPermissions()
        setContent {
            MissedCallSaviourApp()
        }
    }

    private fun checkAndRequestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG
        )
        val missingPermissions = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missingPermissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missingPermissions.toTypedArray(), 101)
        }
    }
}

@Composable
fun MissedCallSaviourApp() {
    val navController = rememberNavController()
    var token by remember { mutableStateOf<String?>(null) }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Background
    ) {
        // Mesh background effect
        Box(modifier = Modifier.fillMaxSize()) {
            CanvasBackground()
            
            if (token == null) {
                LoginScreen(onLoginSuccess = { token = it })
            } else {
                MainScaffold(navController, token!!, onLogout = { token = null })
            }
        }
    }
}

@Composable
fun CanvasBackground() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .drawBehind {
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(PrimaryGradientStart.copy(alpha = 0.15f), Color.Transparent),
                        center = Offset(size.width * 0.2f, size.height * 0.2f),
                        radius = size.width * 0.8f
                    )
                )
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(PrimaryGradientEnd.copy(alpha = 0.1f), Color.Transparent),
                        center = Offset(size.width * 0.8f, size.height * 0.8f),
                        radius = size.width * 0.6f
                    )
                )
            }
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScaffold(navController: NavHostController, token: String, onLogout: () -> Unit) {
    Scaffold(
        bottomBar = { BottomNavigationBar(navController) },
        containerColor = Color.Transparent,
        topBar = {
            CenterAlignedTopAppBar(
                title = { 
                    Text(
                        "SAVIOUR", 
                        letterSpacing = 4.sp,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Black,
                        color = TextPrimary
                    ) 
                },
                actions = {
                    IconButton(onClick = onLogout) {
                        Icon(Icons.Default.Logout, contentDescription = "Logout", tint = ErrorGradientStart)
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(containerColor = Color.Transparent)
            )
        }
    ) { paddingValues ->
        Box(modifier = Modifier.padding(paddingValues)) {
            NavigationHost(navController, token)
        }
    }
}

@Composable
fun NavigationHost(navController: NavHostController, token: String) {
    NavHost(navController, startDestination = "dashboard") {
        composable("dashboard") { DashboardScreen(token) }
        composable("history") { HistoryScreen(token) }
        composable("record") { RecordingScreen(token) }
        composable("chat") { ChatScreen(token) }
    }
}

@Composable
fun DashboardScreen(token: String) {
    var stats by remember { mutableStateOf<DashboardStats?>(null) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) {
        scope.launch {
            try {
                val response = RetrofitClient.instance.getStats("Bearer $token")
                if (response.isSuccessful) {
                    stats = response.body()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
    ) {
        Text(
            "Overview",
            color = TextPrimary,
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            "Real-time call recovery status",
            color = TextSecondary,
            fontSize = 14.sp
        )
        Spacer(modifier = Modifier.height(24.dp))
        
        // Premium Stat Card
        PremiumCard {
            Column(modifier = Modifier.padding(24.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .clip(CircleShape)
                            .background(SuccessGradientStart)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("ESTIMATED REVENUE", color = TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = if (stats != null) "$${stats?.est_revenue}" else "$0.00",
                    style = MaterialTheme.typography.displayMedium.copy(
                        brush = Brush.linearGradient(listOf(PrimaryGradientStart, PrimaryGradientEnd)),
                        fontWeight = FontWeight.Black
                    )
                )
                Spacer(modifier = Modifier.height(20.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    StatItem("Calls Saved", stats?.missed_calls_saved?.toString() ?: "0", PrimaryGradientStart)
                    StatItem("Efficiency", "92%", SuccessGradientStart)
                }
            }
        }
        
        Spacer(modifier = Modifier.height(20.dp))
        
        Text("AI ACTIVITY", color = TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(start = 4.dp))
        Spacer(modifier = Modifier.height(12.dp))
        
        RecentActivityItem("Alex handled +91 98XXX", "2 mins ago", Icons.Default.AutoAwesome)
        RecentActivityItem("New lead captured", "15 mins ago", Icons.Default.AddBusiness)
    }
}

@Composable
fun StatItem(label: String, value: String, color: Color) {
    Column {
        Text(label, color = TextSecondary, fontSize = 12.sp)
        Text(value, color = color, fontSize = 20.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun RecentActivityItem(title: String, time: String, icon: ImageVector) {
    PremiumCard(modifier = Modifier.padding(vertical = 4.dp)) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(CardBlur),
                contentAlignment = Alignment.Center
            ) {
                Icon(icon, contentDescription = null, tint = PrimaryGradientEnd, modifier = Modifier.size(20.dp))
            }
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(title, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Medium)
                Text(time, color = TextSecondary, fontSize = 12.sp)
            }
        }
    }
}

@Composable
fun HistoryScreen(token: String) {
    var logs by remember { mutableStateOf<List<CallLog>>(emptyList()) }
    val scope = rememberCoroutineScope()

    LaunchedEffect(Unit) {
        scope.launch {
            try {
                val response = RetrofitClient.instance.getHistory("Bearer $token")
                if (response.isSuccessful) {
                    logs = response.body() ?: emptyList()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    LazyColumn(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        item { Text("Activity Log", color = TextPrimary, fontSize = 24.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(bottom = 20.dp)) }
        items(logs) { log ->
            PremiumCard(modifier = Modifier.padding(vertical = 6.dp)) {
                Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.PhoneCallback, contentDescription = null, tint = PrimaryGradientStart)
                    Spacer(modifier = Modifier.width(16.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(log.phone_number, color = TextPrimary, fontWeight = FontWeight.Bold)
                        Text(log.summary, color = TextSecondary, fontSize = 12.sp, maxLines = 1)
                    }
                    Text(log.timestamp.take(5), color = TextSecondary, fontSize = 12.sp)
                }
            }
        }
    }
}

data class LogItem(val number: String, val status: String, val time: String, val tag: String)

import android.media.MediaRecorder
import java.io.File
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import android.util.Log

// ... (other imports)

@Composable
fun RecordingScreen(token: String) {
    val context = androidx.compose.ui.platform.LocalContext.current
    var isRecording by remember { mutableStateOf(false) }
    var isUploading by remember { mutableStateOf(false) }
    var recorder by remember { mutableStateOf<MediaRecorder?>(null) }
    var audioFile by remember { mutableStateOf<File?>(null) }
    val scope = rememberCoroutineScope()

    val infiniteTransition = rememberInfiniteTransition()
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        )
    )

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(contentAlignment = Alignment.Center) {
            if (isRecording) {
                Box(
                    modifier = Modifier
                        .size(180.dp)
                        .graphicsLayer { scaleX = pulseScale; scaleY = pulseScale }
                        .background(PrimaryGradientStart.copy(alpha = 0.2f), CircleShape)
                )
            }
            Button(
                onClick = {
                    if (!isRecording) {
                        try {
                            val file = File(context.cacheDir, "recording_${System.currentTimeMillis()}.m4a")
                            audioFile = file
                            recorder = MediaRecorder().apply {
                                setAudioSource(MediaRecorder.AudioSource.MIC)
                                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                                setOutputFile(file.absolutePath)
                                prepare()
                                start()
                            }
                            isRecording = true
                        } catch (e: Exception) {
                            Log.e("Recording", "Start failed", e)
                        }
                    } else {
                        try {
                            recorder?.apply {
                                stop()
                                release()
                            }
                            recorder = null
                            isRecording = false
                            
                            // Upload immediately
                            isUploading = true
                            scope.launch {
                                try {
                                    audioFile?.let { file ->
                                        val reqFile = file.asRequestBody("audio/m4a".toMediaTypeOrNull())
                                        val body = MultipartBody.Part.createFormData("file", file.name, reqFile)
                                        val response = RetrofitClient.instance.uploadRecording("Bearer $token", body)
                                        
                                        withContext(Dispatchers.Main) {
                                            if (response.isSuccessful) {
                                                Log.d("Upload", "Success")
                                            }
                                            isUploading = false
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e("Upload", "Failed", e)
                                    withContext(Dispatchers.Main) { isUploading = false }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("Recording", "Stop failed", e)
                        }
                    }
                },
                modifier = Modifier.size(120.dp),
                enabled = !isUploading,
                shape = CircleShape,
                colors = ButtonDefaults.buttonColors(containerColor = if (isRecording) ErrorGradientStart else PrimaryGradientStart),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 8.dp)
            ) {
                if (isUploading) {
                    CircularProgressIndicator(color = Color.White)
                } else {
                    Icon(
                        if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                        contentDescription = null,
                        modifier = Modifier.size(48.dp)
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(40.dp))
        Text(
            if (isUploading) "ANALYZING AUDIO..." else if (isRecording) "RECORDING IN PROGRESS" else "TAP TO CAPTURE LEAD",
            color = if (isRecording) ErrorGradientEnd else TextPrimary,
            fontWeight = FontWeight.Black,
            letterSpacing = 2.sp
        )
        Text(
            "Speak clearly to summarize the call manually",
            color = TextSecondary,
            fontSize = 12.sp,
            modifier = Modifier.padding(top = 8.dp)
        )
    }
}

@Composable
fun ChatScreen(token: String) {
    var chatInput by remember { mutableStateOf("") }
    val chatMessages = remember { mutableStateListOf<Pair<String, String>>(
        "model" to "Hello! I'm Alex, your AI assistant. How can I help with your business calls today?"
    ) }
    var isTyping by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        LazyColumn(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(chatMessages) { (role, content) ->
                val isModel = role == "model"
                Box(
                    modifier = Modifier.fillMaxWidth(),
                    contentAlignment = if (isModel) Alignment.CenterStart else Alignment.CenterEnd
                ) {
                    PremiumCard(
                        modifier = Modifier.widthIn(max = 280.dp),
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            if (isModel) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Default.AutoAwesome, contentDescription = null, tint = PrimaryGradientEnd, modifier = Modifier.size(12.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("ALEX AI", color = TextSecondary, fontSize = 9.sp, fontWeight = FontWeight.Bold)
                                }
                                Spacer(modifier = Modifier.height(4.dp))
                            }
                            Text(
                                text = content,
                                color = TextPrimary,
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            }
            if (isTyping) {
                item {
                    Text("Alex is thinking...", color = TextSecondary, fontSize = 12.sp, modifier = Modifier.padding(start = 8.dp))
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextField(
                value = chatInput,
                onValueChange = { chatInput = it },
                modifier = Modifier.weight(1f).clip(RoundedCornerShape(24.dp)),
                placeholder = { Text("Command AI Agent...", color = TextSecondary) },
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = Surface,
                    unfocusedContainerColor = Surface,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary
                )
            )
            Spacer(modifier = Modifier.width(12.dp))
            FloatingActionButton(
                onClick = {
                    if (chatInput.isNotBlank()) {
                        val message = chatInput
                        chatMessages.add("user" to message)
                        chatInput = ""
                        isTyping = true
                        
                        scope.launch {
                            try {
                                val response = RetrofitClient.instance.sendChat(
                                    "Bearer $token",
                                    ChatRequest(message, "android-session")
                                )
                                withContext(Dispatchers.Main) {
                                    if (response.isSuccessful) {
                                        chatMessages.add("model" to (response.body()?.reply ?: "No response"))
                                    } else {
                                        chatMessages.add("model" to "Error: Could not reach AI server.")
                                    }
                                    isTyping = false
                                }
                            } catch (e: Exception) {
                                withContext(Dispatchers.Main) {
                                    chatMessages.add("model" to "Network Error: ${e.message}")
                                    isTyping = false
                                }
                            }
                        }
                    }
                },
                containerColor = PrimaryGradientStart,
                shape = CircleShape,
                modifier = Modifier.size(52.dp)
            ) {
                Icon(Icons.Default.Send, contentDescription = null, tint = Color.White)
            }
        }
    }
}

@Composable
fun BottomNavigationBar(navController: NavHostController) {
    val items = listOf(
        Triple("dashboard", "Overview", Icons.Default.Dashboard),
        Triple("history", "Logs", Icons.Default.History),
        Triple("record", "Capture", Icons.Default.Mic),
        Triple("chat", "AI Chat", Icons.Default.Chat)
    )
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = Surface.copy(alpha = 0.9f)),
        elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
    ) {
        NavigationBar(containerColor = Color.Transparent) {
            val navBackStackEntry by navController.currentBackStackEntryAsState()
            val currentRoute = navBackStackEntry?.destination?.route
            items.forEach { (route, label, icon) ->
                val selected = currentRoute == route
                NavigationBarItem(
                    icon = { 
                        Icon(
                            icon, 
                            contentDescription = label,
                            tint = if (selected) PrimaryGradientStart else TextSecondary
                        ) 
                    },
                    label = { 
                        Text(
                            label, 
                            color = if (selected) TextPrimary else TextSecondary,
                            fontSize = 11.sp,
                            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
                        ) 
                    },
                    selected = selected,
                    onClick = {
                        navController.navigate(route) {
                            popUpTo(navController.graph.startDestinationId)
                            launchSingleTop = true
                        }
                    },
                    colors = NavigationBarItemDefaults.colors(indicatorColor = Color.Transparent)
                )
            }
        }
    }
}

@Composable
fun LoginScreen(onLoginSuccess: (String) -> Unit) {
    var email by remember { mutableStateOf("") }
    var pass by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var errorMsg by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(80.dp)
                .background(Brush.linearGradient(listOf(PrimaryGradientStart, PrimaryGradientEnd)), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Shield, contentDescription = null, tint = Color.White, modifier = Modifier.size(40.dp))
        }
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            "SAVIOUR", 
            color = TextPrimary, 
            fontSize = 40.sp, 
            fontWeight = FontWeight.Black,
            letterSpacing = 8.sp
        )
        Text("AI POWERED CALL INTELLIGENCE", color = PrimaryGradientEnd, fontSize = 10.sp, fontWeight = FontWeight.Bold)
        
        Spacer(modifier = Modifier.height(40.dp))
        
        OutlinedTextField(
            value = email, 
            onValueChange = { email = it }, 
            label = { Text("Business Email") },
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = PrimaryGradientStart,
                unfocusedBorderColor = BorderColor,
                focusedLabelColor = PrimaryGradientStart,
                focusedTextColor = TextPrimary,
                unfocusedTextColor = TextPrimary
            )
        )
        
        Spacer(modifier = Modifier.height(16.dp))

        OutlinedTextField(
            value = pass, 
            onValueChange = { pass = it }, 
            label = { Text("Password") },
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = PrimaryGradientStart,
                unfocusedBorderColor = BorderColor,
                focusedLabelColor = PrimaryGradientStart,
                focusedTextColor = TextPrimary,
                unfocusedTextColor = TextPrimary
            )
        )
        
        if (errorMsg != null) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(errorMsg!!, color = ErrorGradientStart, fontSize = 12.sp)
        }

        Spacer(modifier = Modifier.height(32.dp))
        
        Button(
            onClick = { 
                if (email.isNotBlank() && pass.isNotBlank()) {
                    isLoading = true
                    errorMsg = null
                    scope.launch {
                        try {
                            val response = RetrofitClient.instance.login(email, pass)
                            if (response.isSuccessful) {
                                onLoginSuccess(response.body()?.access_token ?: "")
                            } else {
                                errorMsg = "Login Failed: Invalid credentials"
                            }
                        } catch (e: Exception) {
                            errorMsg = "Login Failed: ${e.message}"
                        } finally {
                            isLoading = false
                        }
                    }
                }
            }, 
            modifier = Modifier.fillMaxWidth().height(60.dp),
            enabled = !isLoading,
            colors = ButtonDefaults.buttonColors(containerColor = PrimaryGradientStart),
            shape = RoundedCornerShape(16.dp)
        ) {
            if (isLoading) {
                CircularProgressIndicator(color = Color.White, modifier = Modifier.size(24.dp))
            } else {
                Text("AUTHENTICATE", fontWeight = FontWeight.Black, letterSpacing = 2.sp)
            }
        }
        
        Spacer(modifier = Modifier.height(24.dp))
        Text("Secure access to your enterprise dashboard", color = TextSecondary, fontSize = 12.sp)
    }
}

@Composable
fun PremiumCard(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .border(1.dp, BorderColor, RoundedCornerShape(24.dp)),
        colors = CardDefaults.cardColors(containerColor = CardBlur),
        shape = RoundedCornerShape(24.dp),
        content = { content() }
    )
}

// Add Gradient Text extension for Modifier
fun Modifier.textGradient(colors: List<Color>): Modifier = this.drawBehind {
    // This is a simplified version, usually we'd use style = TextStyle(brush = ...)
}
