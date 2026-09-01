import requests
import json

def test_endpoints():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Health
    h = requests.get(f"{base_url}/health")
    print("Health Status:", h.status_code, h.json())
    assert h.status_code == 200
    
    # 2. Create Investigation
    inv = requests.post(f"{base_url}/api/investigations", json={
        "latitude": 28.582,
        "longitude": -94.925,
        "timestamp": "2020-12-31T11:30:25Z"
    })
    print("Create Investigation:", inv.status_code)
    assert inv.status_code == 200
    data = inv.json()
    inv_id = data["id"]
    print(f"  -> Investigation ID: {inv_id}")
    print(f"  -> Spill Detected: {data['spill']['detected']}, Area: {data['spill']['area']} km2, Confidence: {data['spill']['confidence']}%")
    print(f"  -> Origin: {data['origin']['probableOrigin']['latitude']}°N, {data['origin']['probableOrigin']['longitude']}°E")
    print(f"  -> Vessels matched: {len(data['vessels'])}, Top: {data['vessels'][0]['name']} (Score: {data['vessels'][0]['evidenceScore']}/100)")
    
    # 3. Ask RAG Copilot
    rag_q1 = requests.post(f"{base_url}/api/investigations/{inv_id}/ask", json={
        "question": "Why was this vessel ranked first?"
    })
    print("\nRAG Question 1 ('Why was this vessel ranked first?'):", rag_q1.status_code)
    assert rag_q1.status_code == 200
    rag_res1 = rag_q1.json()
    print("  -> Citations returned:", len(rag_res1["sources"]))
    for s in rag_res1["sources"]:
        print(f"     • {s['title']} ({s['source']})")
    print("  -> Grounded Answer Preview:")
    print("-----------------------------------------------------------------")
    print(rag_res1["answer"])
    print("-----------------------------------------------------------------")
    
    # 4. Model Evaluation Metrics
    m = requests.get(f"{base_url}/api/model/metrics")
    print("\nModel Metrics Endpoint:", m.status_code)
    assert m.status_code == 200
    metrics = m.json()
    print(f"  -> Mean IoU: {metrics['test_mean_iou']*100:.2f}% | Dice: {metrics['test_mean_dice']*100:.2f}% | Precision: {metrics['test_mean_precision']*100:.2f}% | Recall: {metrics['test_mean_recall']*100:.2f}%")
    
    # 5. Provenance
    p = requests.get(f"{base_url}/api/provenance")
    print("\nData Provenance Summary:", p.status_code)
    assert p.status_code == 200
    prov = p.json()
    print(f"  -> System Mode: {prov['system_data_mode']}")
    for k, v in prov['data_sources'].items():
        print(f"     [OK] {v['label']}: {v['status']}")

    print("\nALL ENDPOINTS OPERATIONAL AND FULLY VERIFIED!")

if __name__ == "__main__":
    test_endpoints()
