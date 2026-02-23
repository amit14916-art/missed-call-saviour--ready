import { StatusBar } from 'expo-status-bar';
import {
    StyleSheet, Text, View, TouchableOpacity, TextInput,
    ScrollView, PermissionsAndroid, Platform, Alert, RefreshControl,
    Modal, Animated, Dimensions
} from 'react-native';
import React, { useEffect, useState, useRef } from 'react';
import { WebView } from 'react-native-webview';
import { Audio } from 'expo-av';
import * as Notifications from 'expo-notifications';
import { LinearGradient } from 'expo-linear-gradient';
import {
    Mic, History, LayoutDashboard, Settings,
    Play, Square, RefreshCcw, LogOut,
    TrendingUp, PhoneCall, CheckCircle, Save,
    UserCircle, Bot, MessageSquare, ChevronRight
} from 'lucide-react-native';

const { width } = Dimensions.get('window');

// ─── CONFIG ───────────────────────────────────────────────
// IMPORTANT: Change this to your local IP if testing locally (e.g., http://192.168.0.177:8000)
// For Railway, use: https://missed-call-saviour-ready-production.up.railway.app
const BASE_URL = 'http://192.168.0.177:8000';
const SERVER_URL = `${BASE_URL}/api/upload-call-recording`;

// Show notifications even when app is in foreground
Notifications.setNotificationHandler({
    handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: true,
        shouldSetBadge: false,
    }),
});

// ─── MAIN APP ─────────────────────────────────────────────
export default function App() {
    const [token, setToken] = useState(null);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [activeTab, setActiveTab] = useState('dashboard');
    const [permissionsGranted, setPermissionsGranted] = useState(false);
    const [recording, setRecording] = useState(null);
    const [isRecording, setIsRecording] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [callHistory, setCallHistory] = useState([]);
    const [dashboardStats, setDashboardStats] = useState({ saved: 0, revenue: 0, weekly_volume: [], weekly_labels: [] });
    const [aiConfig, setAiConfig] = useState({ business_name: '', greeting: '', persona: 'friendly' });
    const [refreshing, setRefreshing] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [timerInterval, setTimerInterval] = useState(null);
    const [selectedCall, setSelectedCall] = useState(null);
    const [modalVisible, setModalVisible] = useState(false);
    const [callerName, setCallerName] = useState('');

    // Chat Native States
    const [chatModalVisible, setChatModalVisible] = useState(false);
    const [chatMessages, setChatMessages] = useState([
        { id: 1, text: "Hello! I'm Alex, your AI assistant. How can I help with your business calls today?", sender: 'ai' }
    ]);
    const [chatInput, setChatInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);

    // Animations
    const pulseAnim = useRef(new Animated.Value(1)).current;
    const chatFabAnim = useRef(new Animated.Value(0)).current;
    const chatScrollViewRef = useRef(null);

    useEffect(() => {
        requestPermissions();
        if (token) {
            refreshAllData();
        }
    }, [token]);

    useEffect(() => {
        if (isRecording) {
            Animated.loop(
                Animated.sequence([
                    Animated.timing(pulseAnim, { toValue: 1.2, duration: 800, useNativeDriver: true }),
                    Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true })
                ])
            ).start();
        } else {
            pulseAnim.setValue(1);
        }
    }, [isRecording]);

    const requestPermissions = async () => {
        try {
            if (Platform.OS === 'android') {
                const granted = await PermissionsAndroid.requestMultiple([
                    PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
                ]);
                if (granted['android.permission.RECORD_AUDIO'] === PermissionsAndroid.RESULTS.GRANTED) {
                    setPermissionsGranted(true);
                }
            } else {
                const res = await Audio.requestPermissionsAsync();
                if (res.granted) setPermissionsGranted(true);
            }
            await Notifications.requestPermissionsAsync();
        } catch (err) { console.warn(err); }
    };

    const refreshAllData = () => {
        fetchDashboardStats();
        fetchHistory();
        fetchAiConfig();
    };

    const handleLogin = async () => {
        if (!email || !password) {
            Alert.alert("Required", "Please enter credentials.");
            return;
        }
        try {
            const formData = new URLSearchParams();
            formData.append('username', email);
            formData.append('password', password);

            const response = await fetch(`${BASE_URL}/token`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData.toString(),
            });

            const data = await response.json();
            if (response.ok) {
                setToken(data.access_token);
            } else {
                console.error("Login Failed:", data);
                Alert.alert("Login Failed", data.detail || "Invalid credentials");
            }
        } catch (err) {
            console.error("Network Error:", err);
            Alert.alert("Network Error", "Could not connect to " + BASE_URL);
        }
    };

    const fetchDashboardStats = async () => {
        if (!token) return;
        try {
            const response = await fetch(`${BASE_URL}/api/dashboard/stats`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const data = await response.json();
            if (response.ok) {
                setDashboardStats({
                    saved: data.missed_calls_saved || 0,
                    revenue: data.est_revenue || 0,
                    weekly_volume: data.weekly_volume || [],
                    weekly_labels: data.weekly_labels || []
                });
            }
        } catch (err) { console.error(err); }
    };

    const fetchHistory = async () => {
        if (!token) return;
        setRefreshing(true);
        try {
            const response = await fetch(`${BASE_URL}/api/calls`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const data = await response.json();
            if (response.ok) setCallHistory(data);
        } catch (err) { console.error(err); } finally { setRefreshing(false); }
    };

    const fetchAiConfig = async () => {
        if (!token) return;
        try {
            const response = await fetch(`${BASE_URL}/api/ai-config`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const data = await response.json();
            if (response.ok) setAiConfig(data);
        } catch (err) { console.error(err); }
    };

    const updateAiConfig = async () => {
        if (!token) return;
        try {
            const response = await fetch(`${BASE_URL}/api/ai-config`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(aiConfig)
            });
            if (response.ok) {
                Alert.alert("Success", "AI Assistant Updated!");
            }
        } catch (err) { Alert.alert("Error", "Could not update config."); }
    };

    const sendChatMessage = async () => {
        if (!chatInput.trim()) return;

        const userMsg = { id: Date.now(), text: chatInput, sender: 'user' };
        setChatMessages(prev => [...prev, userMsg]);
        setChatInput('');
        setIsTyping(true);

        try {
            const response = await fetch(`${BASE_URL}/api/analyze-chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    message: userMsg.text,
                    session_id: email
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Chat Server Error:", errorText);
                throw new Error("Offline");
            }

            const data = await response.json();
            const aiMsg = { id: Date.now() + 1, text: data.reply, sender: 'ai' };
            setChatMessages(prev => [...prev, aiMsg]);
        } catch (err) {
            console.error("Chat error:", err);
            Alert.alert("Alex is Offline", "The AI model is currently initializing or deployed on a different branch. Please try again in 2 minutes.");
        } finally {
            setIsTyping(false);
        }
    };

    const reSummarizeCall = async (callId) => {
        if (!token) return;
        try {
            const response = await fetch(`${BASE_URL}/api/calls/${callId}/re-summarize`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                Alert.alert("Processing", "AI is re-analyzing...");
                fetchHistory();
                setModalVisible(false);
            }
        } catch (err) { console.error(err); }
    };

    async function startRecording() {
        if (!token) return;
        try {
            const permission = await Audio.requestPermissionsAsync();
            if (permission.status !== 'granted') {
                Alert.alert("Permission Required", "Please allow microphone access in settings.");
                return;
            }

            await Audio.setAudioModeAsync({
                allowsRecordingIOS: true,
                playsInSilentModeIOS: true,
                staysActiveInBackground: true,
                shouldDuckAndroid: true,
                playThroughEarpieceAndroid: false,
            });

            const { recording } = await Audio.Recording.createAsync(
                Audio.RecordingOptionsPresets.HIGH_QUALITY
            );
            setRecording(recording);
            setIsRecording(true);
            setRecordingTime(0);
            const interval = setInterval(() => setRecordingTime(prev => prev + 1), 1000);
            setTimerInterval(interval);
        } catch (err) {
            console.error("Start recording error:", err);
            Alert.alert("Error", "Recording failed. Check if another app is using mic.");
        }
    }

    async function stopAndUpload() {
        if (!recording) return;
        setIsRecording(false);
        if (timerInterval) clearInterval(timerInterval);
        await recording.stopAndUnloadAsync();
        const uri = recording.getURI();
        setRecording(undefined);
        setUploading(true);

        try {
            const formData = new FormData();
            formData.append('file', { uri, name: `call_${Date.now()}.m4a`, type: 'audio/m4a' });

            const response = await fetch(SERVER_URL, {
                method: 'POST',
                body: formData,
                headers: {
                    'Authorization': `Bearer ${token}`
                },
            });
            if (response.ok) {
                Alert.alert("Success", "Call Uploaded & AI Analysis Started.");
                refreshAllData();
            }
        } catch (e) { Alert.alert("Error", "Upload failed."); } finally { setUploading(false); }
    }

    // ─── LOGIN UI ───────────────────────────────
    if (!token) {
        return (
            <LinearGradient colors={['#020617', '#0f172a']} style={styles.container}>
                <View style={[styles.center, { marginTop: 80 }]}>
                    <View style={styles.logoCircle}>
                        <Mic color="#3b82f6" size={40} />
                    </View>
                    <Text style={styles.heroTitle}>Missed Call Saviour</Text>
                    <Text style={styles.heroSub}>Sync your business AI with your phone</Text>

                    <View style={styles.loginCard}>
                        <TextInput
                            style={styles.authInput}
                            placeholder="Email"
                            placeholderTextColor="#64748b"
                            value={email}
                            onChangeText={setEmail}
                            autoCapitalize="none"
                        />
                        <TextInput
                            style={styles.authInput}
                            placeholder="Password"
                            placeholderTextColor="#64748b"
                            value={password}
                            onChangeText={setPassword}
                            secureTextEntry
                        />
                        <TouchableOpacity style={styles.loginBtn} onPress={handleLogin}>
                            <Text style={styles.loginBtnText}>Login to Dashboard</Text>
                        </TouchableOpacity>
                    </View>
                </View>
                <StatusBar style="light" />
            </LinearGradient>
        );
    }

    // ─── MAIN APP UI ────────────────────────────
    return (
        <View style={styles.container}>
            <LinearGradient colors={['#1e293b', '#020617']} style={styles.headerGradient}>
                <View style={styles.headerContent}>
                    <View>
                        <Text style={styles.welcomeText}>Hello,</Text>
                        <Text style={styles.userEmail}>{email.split('@')[0]}</Text>
                    </View>
                    <TouchableOpacity onPress={() => setToken(null)} style={styles.logoutIcon}>
                        <LogOut size={20} color="#94a3b8" />
                    </TouchableOpacity>
                </View>

                {activeTab === 'dashboard' && (
                    <View style={styles.quickStatsRow}>
                        <View style={styles.quickStat}>
                            <TrendingUp size={24} color="#10b981" />
                            <Text style={styles.quickStatValue}>${dashboardStats.revenue}</Text>
                            <Text style={styles.quickStatLabel}>Revenue</Text>
                        </View>
                        <View style={styles.quickStat}>
                            <PhoneCall size={24} color="#3b82f6" />
                            <Text style={styles.quickStatValue}>{dashboardStats.saved}</Text>
                            <Text style={styles.quickStatLabel}>Saved</Text>
                        </View>
                    </View>
                )}
            </LinearGradient>

            <View style={{ flex: 1, backgroundColor: '#020617' }}>
                {/* ── DASHBOARD TAB ── */}
                {activeTab === 'dashboard' && (
                    <ScrollView contentContainerStyle={styles.scrollPadding} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={refreshAllData} tintColor="#3b82f6" />}>
                        <Text style={styles.sectionTitle}>Performance Overview</Text>
                        <View style={styles.bigCard}>
                            <Text style={styles.bigCardLabel}>Estimated Revenue Recovered</Text>
                            <Text style={styles.bigCardValue}>${dashboardStats.revenue}</Text>
                            <View style={styles.progressBarBg}>
                                <View style={[styles.progressBar, { width: '70%' }]} />
                            </View>
                            <Text style={styles.progressText}>70% of target reached ($10k goal)</Text>
                        </View>

                        {/* ── LARGE CHAT WINDOW ── */}
                        <Text style={[styles.sectionTitle, { marginTop: 24 }]}>AI Assistant Chat</Text>
                        <View style={styles.largeChatContainer}>
                            <View style={styles.largeChatHeader}>
                                <Bot color="#3b82f6" size={20} />
                                <Text style={styles.largeChatHeaderText}>Alex is Online</Text>
                            </View>
                            <ScrollView
                                style={styles.largeChatList}
                                nestedScrollEnabled={true}
                                contentContainerStyle={{ padding: 15 }}
                                ref={chatScrollViewRef}
                                onContentSizeChange={() => chatScrollViewRef.current?.scrollToEnd({ animated: true })}
                            >
                                {chatMessages.map(msg => (
                                    <View key={msg.id} style={[styles.msgBubble, msg.sender === 'user' ? styles.msgUser : styles.msgAi, { padding: 10, borderRadius: 12 }]}>
                                        <Text style={[styles.msgText, msg.sender === 'user' ? styles.msgTextUser : styles.msgTextAi, { fontSize: 13 }]}>{msg.text}</Text>
                                    </View>
                                ))}
                                {isTyping && <Text style={styles.typingText}>thinking...</Text>}
                            </ScrollView>
                            <View style={styles.largeChatInputRow}>
                                <TextInput
                                    style={styles.largeChatInput}
                                    placeholder="Type a message..."
                                    placeholderTextColor="#64748b"
                                    value={chatInput}
                                    onChangeText={setChatInput}
                                />
                                <TouchableOpacity onPress={sendChatMessage} style={styles.largeChatSendBtn}>
                                    <ChevronRight color="#3b82f6" size={20} />
                                </TouchableOpacity>
                            </View>
                        </View>

                        <Text style={[styles.sectionTitle, { marginTop: 24 }]}>Recent Saved Leads</Text>
                        {callHistory.slice(0, 3).map((call, i) => (
                            <TouchableOpacity key={i} style={styles.miniCallItem} onPress={() => { setSelectedCall(call); setModalVisible(true); }}>
                                <View style={styles.miniCallIcon}>
                                    <UserCircle size={20} color="#3b82f6" />
                                </View>
                                <View style={{ flex: 1 }}>
                                    <Text style={styles.miniCallPhone}>{call.phone_number}</Text>
                                    <Text style={styles.miniCallSummary} numberOfLines={1}>{call.summary}</Text>
                                </View>
                                <ChevronRight size={16} color="#475569" />
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                )}

                {/* ── RECORD TAB ── */}
                {activeTab === 'record' && (
                    <View style={[styles.scrollPadding, { alignItems: 'center', justifyContent: 'center', flex: 1 }]}>
                        <Animated.View style={[styles.recordPulse, { transform: [{ scale: pulseAnim }] }]}>
                            <TouchableOpacity
                                style={[styles.mainRecordBtn, isRecording && styles.stopBtn]}
                                onPress={isRecording ? stopAndUpload : startRecording}
                                disabled={uploading}
                            >
                                {uploading ? <RefreshCcw size={48} color="#fff" /> :
                                    isRecording ? <Square size={48} color="#fff" fill="#fff" /> :
                                        <Mic size={48} color="#fff" />}
                            </TouchableOpacity>
                        </Animated.View>

                        <Text style={styles.recordStatus}>
                            {uploading ? "Analyzing with AI..." : isRecording ? "LIVE RECORDING" : "Ready to Record"}
                        </Text>
                        {isRecording && (
                            <Text style={styles.recordTimer}>
                                {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
                            </Text>
                        )}

                        {!isRecording && !uploading && (
                            <View style={styles.tipBox}>
                                <Text style={styles.tipText}>💡 Tip: Keep the call on speaker for best AI transcription.</Text>
                            </View>
                        )}
                    </View>
                )}

                {/* ── HISTORY TAB ── */}
                {activeTab === 'history' && (
                    <ScrollView contentContainerStyle={styles.scrollPadding} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={fetchHistory} tintColor="#3b82f6" />}>
                        <Text style={styles.sectionTitle}>Call Logs</Text>
                        {callHistory.length === 0 ? (
                            <View style={styles.emptyCenter}>
                                <History size={64} color="#1e293b" />
                                <Text style={styles.emptyText}>No logs found.</Text>
                            </View>
                        ) : (
                            callHistory.map((call, i) => (
                                <TouchableOpacity key={i} style={styles.historyCard} onPress={() => { setSelectedCall(call); setModalVisible(true); }}>
                                    <View style={styles.historyHeader}>
                                        <Text style={styles.historyPhone}>{call.phone_number}</Text>
                                        <Text style={styles.historyDate}>{new Date(call.timestamp).toLocaleDateString()}</Text>
                                    </View>
                                    <Text style={styles.historySummary} numberOfLines={2}>{call.summary}</Text>
                                    <View style={styles.historyFooter}>
                                        <View style={styles.tag}><Text style={styles.tagText}>AI Summary</Text></View>
                                        {call.recording_url && <View style={[styles.tag, { backgroundColor: '#3b82f622' }]}><Text style={[styles.tagText, { color: '#3b82f6' }]}>Audio Available</Text></View>}
                                    </View>
                                </TouchableOpacity>
                            ))
                        )}
                    </ScrollView>
                )}

                {/* ── SETTINGS TAB ── */}
                {activeTab === 'settings' && (
                    <ScrollView contentContainerStyle={styles.scrollPadding}>
                        <Text style={styles.sectionTitle}>AI Agent Settings</Text>
                        <View style={styles.configCard}>
                            <Text style={styles.fieldLabel}>Business Name</Text>
                            <TextInput
                                style={styles.fieldInput}
                                value={aiConfig.business_name}
                                onChangeText={v => setAiConfig({ ...aiConfig, business_name: v })}
                            />

                            <Text style={[styles.fieldLabel, { marginTop: 16 }]}>Greeting Message</Text>
                            <TextInput
                                style={[styles.fieldInput, { height: 80 }]}
                                multiline
                                value={aiConfig.greeting}
                                onChangeText={v => setAiConfig({ ...aiConfig, greeting: v })}
                            />

                            <Text style={[styles.fieldLabel, { marginTop: 16 }]}>Voice Persona</Text>
                            <View style={styles.personaRow}>
                                {['friendly', 'professional', 'urgent'].map(p => (
                                    <TouchableOpacity
                                        key={p}
                                        style={[styles.personaBtn, aiConfig.persona === p && styles.personaBtnActive]}
                                        onPress={() => setAiConfig({ ...aiConfig, persona: p })}
                                    >
                                        <Text style={[styles.personaBtnText, aiConfig.persona === p && styles.personaBtnTextActive]}>
                                            {p.toUpperCase()}
                                        </Text>
                                    </TouchableOpacity>
                                ))}
                            </View>

                            <TouchableOpacity style={styles.saveBtn} onPress={updateAiConfig}>
                                <Save size={20} color="#fff" />
                                <Text style={styles.saveBtnText}>Update AI Agent</Text>
                            </TouchableOpacity>
                        </View>

                        <Text style={[styles.sectionTitle, { marginTop: 24 }]}>Advanced</Text>
                        <TouchableOpacity style={styles.actionRow} onPress={() => setActiveTab('webview-dash')}>
                            <LayoutDashboard size={20} color="#94a3b8" />
                            <Text style={styles.actionText}>Open Web Dashboard</Text>
                            <ChevronRight size={16} color="#475569" />
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.actionRow} onPress={() => setActiveTab('webview-chat')}>
                            <MessageSquare size={20} color="#94a3b8" />
                            <Text style={styles.actionText}>Open AI Chat</Text>
                            <ChevronRight size={16} color="#475569" />
                        </TouchableOpacity>
                    </ScrollView>
                )}

                {/* ── WEBVIEW TABS ── */}
                {activeTab === 'webview-dash' && <WebView source={{ uri: `${BASE_URL}/dashboard` }} style={{ flex: 1 }} />}
                {activeTab === 'webview-chat' && <WebView source={{ uri: `${BASE_URL}/chat_only` }} style={{ flex: 1 }} />}
            </View>

            {/* ── CHAT FAB ── */}
            <TouchableOpacity
                style={styles.chatFab}
                onPress={() => setChatModalVisible(true)}
            >
                <MessageSquare color="#fff" size={28} />
                <View style={styles.chatFabBadge} />
            </TouchableOpacity>

            {/* ── BOTTOM NAV ── */}
            <View style={styles.bottomNav}>
                {[
                    { id: 'dashboard', icon: LayoutDashboard, label: 'Dash' },
                    { id: 'history', icon: History, label: 'Logs' },
                    { id: 'record', icon: Mic, label: 'Record' },
                    { id: 'settings', icon: Settings, label: 'Config' },
                ].map(item => (
                    <TouchableOpacity key={item.id} style={styles.navItem} onPress={() => setActiveTab(item.id)}>
                        <item.icon size={22} color={activeTab === item.id ? '#3b82f6' : '#64748b'} />
                        <Text style={[styles.navText, activeTab === item.id && styles.navTextActive]}>{item.label}</Text>
                    </TouchableOpacity>
                ))}
            </View>

            {/* ── DETAIL MODAL ── */}
            <Modal animationType="slide" transparent={true} visible={modalVisible} onRequestClose={() => setModalVisible(false)}>
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <View style={styles.modalHeader}>
                            <Text style={styles.modalTitle}>Call Insight</Text>
                            <TouchableOpacity onPress={() => setModalVisible(false)}><Text style={styles.modalClose}>✕</Text></TouchableOpacity>
                        </View>
                        <ScrollView style={{ flex: 1 }}>
                            <View style={styles.modalLeadInfo}>
                                <UserCircle size={40} color="#3b82f6" />
                                <View style={{ marginLeft: 12 }}>
                                    <Text style={styles.modalLeadPhone}>{selectedCall?.phone_number}</Text>
                                    <Text style={styles.modalLeadDate}>{new Date(selectedCall?.timestamp).toLocaleString()}</Text>
                                </View>
                            </View>
                            <Text style={styles.modalSectionLabel}>AI SUMMARY</Text>
                            <Text style={styles.modalSummaryText}>{selectedCall?.summary || "Analyzing..."}</Text>
                            <Text style={[styles.modalSectionLabel, { marginTop: 20 }]}>FULL TRANSCRIPT</Text>
                            <Text style={styles.modalTranscriptText}>{selectedCall?.transcript || "No transcript available."}</Text>

                            <TouchableOpacity style={styles.reAnalyzeBtn} onPress={() => reSummarizeCall(selectedCall?.id)}>
                                <Bot size={18} color="#10b981" />
                                <Text style={styles.reAnalyzeText}>Re-analyze with Gemini</Text>
                            </TouchableOpacity>
                        </ScrollView>
                    </View>
                </View>
            </Modal>

            {/* ── NATIVE CHAT MODAL ── */}
            <Modal
                animationType="slide"
                transparent={true}
                visible={chatModalVisible}
                onRequestClose={() => setChatModalVisible(false)}
            >
                <View style={styles.modalOverlay}>
                    <View style={styles.chatModalContent}>
                        <View style={styles.chatHeader}>
                            <View style={styles.chatHeaderLeft}>
                                <View style={styles.avatarMini}>
                                    <Bot color="#3b82f6" size={18} />
                                </View>
                                <View>
                                    <Text style={styles.chatTitle}>Alex AI</Text>
                                    <Text style={styles.chatStatus}>Always Active</Text>
                                </View>
                            </View>
                            <TouchableOpacity onPress={() => setChatModalVisible(false)}>
                                <Text style={styles.modalClose}>✕</Text>
                            </TouchableOpacity>
                        </View>

                        <ScrollView
                            style={styles.chatList}
                            contentContainerStyle={{ paddingVertical: 20 }}
                            ref={chatScrollViewRef}
                            onContentSizeChange={() => chatScrollViewRef.current?.scrollToEnd({ animated: true })}
                        >
                            {chatMessages.map(msg => (
                                <View
                                    key={msg.id}
                                    style={[
                                        styles.msgBubble,
                                        msg.sender === 'user' ? styles.msgUser : styles.msgAi
                                    ]}
                                >
                                    <Text style={[
                                        styles.msgText,
                                        msg.sender === 'user' ? styles.msgTextUser : styles.msgTextAi
                                    ]}>
                                        {msg.text}
                                    </Text>
                                </View>
                            ))}
                            {isTyping && (
                                <View style={styles.msgAi}>
                                    <Text style={styles.typingText}>Alex is thinking...</Text>
                                </View>
                            )}
                        </ScrollView>

                        <View style={styles.chatInputRow}>
                            <TextInput
                                style={styles.chatTextInput}
                                placeholder="Ask Alex anything..."
                                placeholderTextColor="#64748b"
                                value={chatInput}
                                onChangeText={setChatInput}
                                multiline
                            />
                            <TouchableOpacity
                                style={[styles.chatSendBtn, !chatInput.trim() && { opacity: 0.5 }]}
                                onPress={sendChatMessage}
                                disabled={!chatInput.trim()}
                            >
                                <LinearGradient colors={['#3b82f6', '#2563eb']} style={styles.sendIconBg}>
                                    <ChevronRight color="#fff" size={24} />
                                </LinearGradient>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            </Modal>

            <StatusBar style="light" />
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#020617' },
    center: { alignItems: 'center', justifyContent: 'center', padding: 24 },

    // Auth UI
    logoCircle: { width: 80, height: 80, borderRadius: 40, backgroundColor: '#3b82f622', alignItems: 'center', justifyContent: 'center', marginBottom: 20 },
    heroTitle: { color: '#fff', fontSize: 28, fontWeight: 'bold' },
    heroSub: { color: '#94a3b8', fontSize: 16, marginTop: 8, textAlign: 'center' },
    loginCard: { width: '100%', backgroundColor: '#0f172a', padding: 24, borderRadius: 20, marginTop: 40 },
    authInput: { backgroundColor: '#1e293b', color: '#fff', padding: 15, borderRadius: 12, marginBottom: 12, borderWidth: 1, borderColor: '#334155' },
    loginBtn: { backgroundColor: '#3b82f6', padding: 16, borderRadius: 12, alignItems: 'center', marginTop: 8 },
    loginBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },

    // Header
    headerGradient: { paddingTop: 60, paddingBottom: 20, paddingHorizontal: 20, borderBottomLeftRadius: 30, borderBottomRightRadius: 30 },
    headerContent: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    welcomeText: { color: '#94a3b8', fontSize: 14 },
    userEmail: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
    logoutIcon: { width: 36, height: 36, borderRadius: 18, backgroundColor: '#ffffff11', alignItems: 'center', justifyContent: 'center' },
    quickStatsRow: { flexDirection: 'row', justifyContent: 'space-around', marginTop: 24 },
    quickStat: { alignItems: 'center' },
    quickStatValue: { color: '#fff', fontSize: 18, fontWeight: 'bold', marginTop: 4 },
    quickStatLabel: { color: '#94a3b8', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1 },

    // Tabs / Content
    scrollPadding: { padding: 20 },
    sectionTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold', marginBottom: 16 },
    bigCard: { backgroundColor: '#1e293b', padding: 24, borderRadius: 24, borderWidth: 1, borderColor: '#334155' },
    bigCardLabel: { color: '#94a3b8', fontSize: 14 },
    bigCardValue: { color: '#fff', fontSize: 42, fontWeight: 'bold', marginVertical: 8 },
    progressBarBg: { height: 8, backgroundColor: '#0f172a', borderRadius: 4, marginTop: 12 },
    progressBar: { height: 8, backgroundColor: '#10b981', borderRadius: 4 },
    progressText: { color: '#64748b', fontSize: 12, marginTop: 8 },

    // Graph
    graphContainer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', height: 120, backgroundColor: '#0f172a', borderRadius: 20, padding: 16 },
    graphCol: { alignItems: 'center', flex: 1 },
    graphBar: { width: 12, backgroundColor: '#3b82f6', borderRadius: 6 },
    graphLabel: { color: '#475569', fontSize: 10, marginTop: 8 },

    // Mini items
    miniCallItem: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#0f172a', padding: 12, borderRadius: 16, marginBottom: 8 },
    miniCallIcon: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#3b82f611', alignItems: 'center', justifyContent: 'center', marginRight: 12 },
    miniCallPhone: { color: '#fff', fontWeight: '600', fontSize: 14 },
    miniCallSummary: { color: '#64748b', fontSize: 12 },

    // History
    historyCard: { backgroundColor: '#0f172a', padding: 16, borderRadius: 20, marginBottom: 12, borderWidth: 1, borderColor: '#1e293b' },
    historyHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
    historyPhone: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
    historyDate: { color: '#475569', fontSize: 12 },
    historySummary: { color: '#94a3b8', fontSize: 14, lineHeight: 20 },
    historyFooter: { flexDirection: 'row', gap: 8, marginTop: 12 },
    tag: { backgroundColor: '#10b98111', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 6 },
    tagText: { color: '#10b981', fontSize: 10, fontWeight: 'bold' },

    // Recording
    recordPulse: { width: 150, height: 150, borderRadius: 75, backgroundColor: '#3b82f611', alignItems: 'center', justifyContent: 'center' },
    mainRecordBtn: { width: 100, height: 100, borderRadius: 50, backgroundColor: '#3b82f6', alignItems: 'center', justifyContent: 'center', elevation: 12 },
    stopBtn: { backgroundColor: '#ef4444' },
    recordStatus: { color: '#fff', fontSize: 20, fontWeight: 'bold', marginTop: 30 },
    recordTimer: { color: '#ef4444', fontSize: 32, fontWeight: 'bold', marginTop: 10 },
    tipBox: { backgroundColor: '#1e293b', padding: 16, borderRadius: 12, marginTop: 60, width: '90%' },
    tipText: { color: '#94a3b8', fontSize: 14, textAlign: 'center' },

    // Config
    configCard: { backgroundColor: '#0f172a', padding: 20, borderRadius: 20, borderWidth: 1, borderColor: '#1e293b' },
    fieldLabel: { color: '#94a3b8', fontSize: 12, fontWeight: 'bold', letterSpacing: 1 },
    fieldInput: { backgroundColor: '#1e293b', color: '#fff', padding: 15, borderRadius: 12, marginTop: 8, borderWidth: 1, borderColor: '#334155' },
    personaRow: { flexDirection: 'row', gap: 8, marginTop: 12 },
    personaBtn: { flex: 1, padding: 10, backgroundColor: '#1e293b', borderRadius: 8, alignItems: 'center' },
    personaBtnActive: { backgroundColor: '#3b82f6' },
    personaBtnText: { color: '#64748b', fontSize: 10, fontWeight: 'bold' },
    personaBtnTextActive: { color: '#fff' },
    saveBtn: { backgroundColor: '#10b981', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 16, borderRadius: 12, marginTop: 24, gap: 10 },
    saveBtnText: { color: '#fff', fontWeight: 'bold' },
    actionRow: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#0f172a', padding: 16, borderRadius: 16, marginBottom: 8 },
    actionText: { flex: 1, color: '#fff', marginLeft: 12, fontSize: 15 },

    // Bottom Nav
    bottomNav: { flexDirection: 'row', height: 80, backgroundColor: '#0f172a', borderTopWidth: 1, borderTopColor: '#1e293b', paddingBottom: 20 },
    navItem: { flex: 1, alignItems: 'center', justifyContent: 'center' },
    navText: { fontSize: 10, color: '#64748b', marginTop: 4 },
    navTextActive: { color: '#3b82f6', fontWeight: 'bold' },

    // Modal
    modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.9)', justifyContent: 'flex-end' },
    modalContent: { height: '85%', backgroundColor: '#020617', borderTopLeftRadius: 32, borderTopRightRadius: 32, padding: 24 },
    modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 },
    modalTitle: { color: '#fff', fontSize: 24, fontWeight: 'bold' },
    modalClose: { color: '#94a3b8', fontSize: 24 },
    modalLeadInfo: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#0f172a', padding: 16, borderRadius: 20, marginBottom: 24 },
    modalLeadPhone: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
    modalLeadDate: { color: '#64748b', fontSize: 12 },
    modalSectionLabel: { color: '#3b82f6', fontSize: 12, fontWeight: 'bold', letterSpacing: 1 },
    modalSummaryText: { color: '#fff', fontSize: 16, lineHeight: 24, marginTop: 10 },
    modalTranscriptText: { color: '#94a3b8', fontSize: 14, lineHeight: 22, marginTop: 10, fontStyle: 'italic' },
    reAnalyzeBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 16, borderRadius: 16, backgroundColor: '#10b98111', borderWidth: 1, borderColor: '#10b98133', marginTop: 32, marginBottom: 40, gap: 10 },
    reAnalyzeText: { color: '#10b981', fontWeight: 'bold' },

    emptyCenter: { alignItems: 'center', justifyContent: 'center', paddingVertical: 100 },
    emptyText: { color: '#475569', fontSize: 16, marginTop: 16 },

    // Chat Styles
    chatFab: { position: 'absolute', bottom: 100, right: 20, width: 60, height: 60, borderRadius: 30, backgroundColor: '#3b82f6', alignItems: 'center', justifyContent: 'center', elevation: 8, shadowColor: '#3b82f6', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 5 },
    chatFabBadge: { position: 'absolute', top: 0, right: 0, width: 14, height: 14, borderRadius: 7, backgroundColor: '#10b981', borderWidth: 2, borderColor: '#fff' },
    chatModalContent: { height: '80%', backgroundColor: '#0f172a', borderTopLeftRadius: 30, borderTopRightRadius: 30, overflow: 'hidden' },
    chatHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 20, borderBottomWidth: 1, borderBottomColor: '#1e293b', backgroundColor: '#1e293b' },
    chatHeaderLeft: { flexDirection: 'row', alignItems: 'center' },
    avatarMini: { width: 36, height: 36, borderRadius: 18, backgroundColor: '#3b82f622', alignItems: 'center', justifyContent: 'center', marginRight: 12 },
    chatTitle: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
    chatStatus: { color: '#10b981', fontSize: 10 },
    chatList: { flex: 1, paddingHorizontal: 20 },
    msgBubble: { maxWidth: '85%', padding: 14, borderRadius: 18, marginBottom: 10 },
    msgAi: { alignSelf: 'flex-start', backgroundColor: '#1e293b', borderTopLeftRadius: 4 },
    msgUser: { alignSelf: 'flex-end', backgroundColor: '#3b82f6', borderTopRightRadius: 4 },
    msgText: { fontSize: 15, lineHeight: 22 },
    msgTextAi: { color: '#f1f5f9' },
    msgTextUser: { color: '#fff' },
    typingText: { color: '#64748b', fontSize: 13, fontStyle: 'italic' },
    chatInputRow: { flexDirection: 'row', alignItems: 'center', padding: 16, backgroundColor: '#0f172a', borderTopWidth: 1, borderTopColor: '#1e293b' },
    chatTextInput: { flex: 1, backgroundColor: '#1e293b', color: '#fff', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 24, fontSize: 15, maxHeight: 100 },
    chatSendBtn: { marginLeft: 12 },
    sendIconBg: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },

    // Large Chat in Dashboard
    largeChatContainer: { backgroundColor: '#0f172a', borderRadius: 24, height: 450, borderWidth: 1, borderColor: '#1e293b', overflow: 'hidden', marginBottom: 20 },
    largeChatHeader: { flexDirection: 'row', alignItems: 'center', padding: 15, backgroundColor: '#1e293b', borderBottomWidth: 1, borderBottomColor: '#334155', gap: 10 },
    largeChatHeaderText: { color: '#fff', fontSize: 14, fontWeight: 'bold' },
    largeChatList: { flex: 1, backgroundColor: '#020617' },
    largeChatInputRow: { flexDirection: 'row', alignItems: 'center', padding: 15, backgroundColor: '#0f172a', borderTopWidth: 1, borderTopColor: '#1e293b' },
    largeChatInput: { flex: 1, color: '#fff', fontSize: 15, paddingHorizontal: 15, backgroundColor: '#1e293b', borderRadius: 20, height: 44 },
    largeChatSendBtn: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#3b82f6', alignItems: 'center', justifyContent: 'center', marginLeft: 10 }
});
