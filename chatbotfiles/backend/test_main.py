from appv2 import app
from fastapi.testclient import TestClient


client = TestClient(app)


def testmain():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "hello"
def testlogin():
    data = {
        "username": "test@email.com",
        "password": "test"
    }
    response = client.post("/login/",
                           data = data)
    assert response.status_code == 200
    assert  "access_token" in response.json()
def testcreateinvaliduser():
    data = {
        "email": "test@email.com",
        "password": "test",
        "id" : None
    }
    response = client.post("/signup/", json = data)
    print(response.json())
    print(response)
    assert response.status_code == 400
def testcreatuser():
    data = {
        "email": "something new 2",
        "password": "something new 2",
        "id" : "some"
    }
    response = client.post("/signup/", json = data)
    assert response.status_code == 200
def testgenerate():
    data = {
        "username": "test@email.com",
        "password": "test"
    }
    response1 = client.post("/login/",
                           data = data)
    if "access_token" in response1.json():
        headers = {"Authorization": f"Bearer {response1.json()['access_token']}"}
        response2 = client.post("/generate/", json={"input": "What is Deltek"}, headers=headers)
        assert response2.status_code == 200
    

    

