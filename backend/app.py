

from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_engine import ChatbotEngine
import sqlite3
from datetime import datetime
import json
import uuid

app = Flask(__name__)
CORS(app) 

# Initialize chatbot engine
chatbot = ChatbotEngine()

# Database setup for analytics
def init_db():
    """
    SQLite database for storing conversations and analytics
    """
    conn = sqlite3.connect('chatbot_data.db')
    c = conn.cursor()
    
    # Conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            intent TEXT,
            confidence REAL,
            sentiment TEXT,
            needs_escalation BOOLEAN,
            timestamp TEXT
        )
    ''')
    
    # Analytics table
    c.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT,
            metric_value TEXT,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

def log_conversation(session_id, user_msg, bot_resp, intent, confidence, sentiment, escalation):
    """
    Save conversation to database for analytics
    """
    conn = sqlite3.connect('chatbot_data.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations 
        (session_id, user_message, bot_response, intent, confidence, sentiment, needs_escalation, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, user_msg, bot_resp, intent, confidence, 
        json.dumps(sentiment), escalation, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

# ==================== CHAT ENDPOINTS ====================

@app.route('/api/chat', methods=['POST'])
def chat():
   
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Process message through chatbot engine
        result = chatbot.process_message(user_message, session_id)

        # Log to database
        log_conversation(
            session_id,
            user_message,
            result['response'],
            result['intent'],
            result['confidence'],
            result['sentiment'],
            result['needs_escalation']
        )
        
        return jsonify({
            "session_id": session_id,
            "response": result['response'],
            "intent": result['intent'],
            "confidence": result['confidence'],
            "sentiment": result['sentiment']['emotion'],
            "needs_escalation": result['needs_escalation']
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """
    Retrieve conversation history for a session
    """
    try:
        conn = sqlite3.connect('chatbot_data.db')
        c = conn.cursor()
        c.execute('''
            SELECT user_message, bot_response, intent, confidence, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC
        ''', (session_id,))
        
        rows = c.fetchall()
        conn.close()
        
        history = [
            {
                "user_message": row[0],
                "bot_response": row[1],
                "intent": row[2],
                "confidence": row[3],
                "timestamp": row[4]
            }
            for row in rows
        ]
        
        return jsonify({"history": history})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== ADMIN PANEL ENDPOINTS ====================

@app.route('/api/admin/intents', methods=['GET'])
def get_intents():
    """
    Get all available intents for admin management
    """
    try:
        return jsonify({
            "intents": chatbot.intents_db
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/intents', methods=['POST'])
def add_intent():
    """
    Add new intent through admin panel
    Expected: {"tag": "new_intent", "patterns": [...], "responses": [...]}
    """
    try:
        data = request.json
        tag = data.get('tag')
        patterns = data.get('patterns', [])
        responses = data.get('responses', [])
        
        if not tag or not patterns or not responses:
            return jsonify({"error": "Missing required fields"}), 400
        
        success = chatbot.add_intent(tag, patterns, responses)
        
        return jsonify({
            "success": success,
            "message": f"Intent '{tag}' added successfully"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/analytics', methods=['GET'])
def get_analytics():
    """
    Get comprehensive analytics for dashboard
    """
    try:
        conn = sqlite3.connect('chatbot_data.db')
        c = conn.cursor()
        
        # Total conversations
        c.execute('SELECT COUNT(*) FROM conversations')
        total_convos = c.fetchone()[0]
        
        # Intent distribution
        c.execute('''
            SELECT intent, COUNT(*) as count 
            FROM conversations 
            GROUP BY intent 
            ORDER BY count DESC
            LIMIT 10
        ''')
        intent_dist = [{"intent": row[0], "count": row[1]} for row in c.fetchall()]
        
        # Average confidence
        c.execute('SELECT AVG(confidence) FROM conversations')
        avg_confidence = c.fetchone()[0] or 0
        
        # Sentiment distribution
        c.execute('''
            SELECT sentiment, COUNT(*) as count 
            FROM conversations 
            GROUP BY sentiment
        ''')
        sentiment_dist = [{"sentiment": row[0], "count": row[1]} for row in c.fetchall()]
        
        # Escalation rate
        c.execute('SELECT COUNT(*) FROM conversations WHERE needs_escalation = 1')
        escalations = c.fetchone()[0]
        escalation_rate = (escalations / total_convos * 100) if total_convos > 0 else 0
        
        # Recent unresolved queries
        c.execute('''
            SELECT user_message, intent, confidence, timestamp
            FROM conversations
            WHERE confidence < 0.5 OR needs_escalation = 1
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        unresolved = [
            {
                "query": row[0],
                "intent": row[1],
                "confidence": row[2],
                "timestamp": row[3]
            }
            for row in c.fetchall()
        ]
        
        conn.close()
        
        return jsonify({
            "total_conversations": total_convos,
            "intent_distribution": intent_dist,
            "average_confidence": round(avg_confidence, 3),
            "sentiment_distribution": sentiment_dist,
            "escalation_rate": round(escalation_rate, 2),
            "unresolved_queries": unresolved,
            "active_intents": len(chatbot.intents_db)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/performance', methods=['GET'])
def get_performance():
    """
    Get performance metrics over time
    """
    try:
        conn = sqlite3.connect('chatbot_data.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT confidence, timestamp 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        confidence_trend = [
            {"confidence": row[0], "timestamp": row[1]} 
            for row in c.fetchall()
        ]
        
        conn.close()
        
        return jsonify({
            "confidence_trend": confidence_trend
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Check if API and chatbot are operational
    """
    return jsonify({
        "status": "operational",
        "chatbot_loaded": chatbot is not None,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print(" Server running at: http://localhost:5000")
    print(" Admin panel: Open index.html in browser")
    print(" Chat API: POST to /api/chat")
    print("\nPress CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)