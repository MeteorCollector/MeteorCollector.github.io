var prename = ''

document.getElementById('messageForm').addEventListener('submit', function(event) {
  event.preventDefault();
  
  console.log("Sending...")
  var nameInput = document.getElementById('nameInput').value;
  var messageInput = document.getElementById('messageInput').value.trim();
  
  if (nameInput === '') {
      alert('You haven\'t entered your name!');
      return;
  }
  
  if (messageInput === '') {
      alert('You can not send empty message.');
      return;
  }

  prename = nameInput;
  
  var data = {
      msg: messageInput,
      chid: 0,
      type: 'text',
      filename: '',
      uname: nameInput,
      uid: generateHash(nameInput)
  };

  sendDataToServer(data);
});

function generateHash(name) {
  // get hashed uid
  var hash = 0;
  for (var i = 0; i < name.length; i++) {
      hash = (hash << 5) - hash + name.charCodeAt(i);
  }
  return Math.abs(hash % 10000000000);
}

function sendDataToServer(data) {
  var xhr = new XMLHttpRequest();
  xhr.open('POST', 'http://localhost:5002', true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  
  xhr.onreadystatechange = function() {
      if (xhr.readyState === 4 && xhr.status === 200) {
          console.log('message sent');
      }
  };
  xhr.send(JSON.stringify(data));

  
}