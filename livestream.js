var prename = ''

document.getElementById('messageForm').addEventListener('submit', function(event) {
  event.preventDefault();
  
  console.log("Reading input...")
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
  
  var json_data = {
      msg: messageInput,
      chid: 0,
      type: 'text',
      filename: '',
      uname: nameInput,
      uid: generateHash(nameInput)
  };

  sendDataToServer(json_data);
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
  console.log("Connecting to server...")
  var xhr = new XMLHttpRequest();
  xhr.open('POST', 'http://120.46.209.170:5002', true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(JSON.stringify(data));
  
  xhr.onreadystatechange = function() {
      console.log("Sending...")
      if (xhr.readyState === 4 && xhr.status === 200) {
          console.log('message sent');
      }
  };
  //xhr.send(JSON.stringify(data));
  document.getElementById('messageInput').value = '';
  
}


// function sendDataToServer(json_data) {
//   console.log("Connecting to server...")
//   fetch('http://120.46.209.170:5002', {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json'
//     },
//     body: JSON.stringify(json_data)
//   })
//   .then(response => {
//     if (!response.ok) {
//       throw new Error('Network response was not ok');
//     }
//     return response.json();
//   })
//   .then(data => {
//     console.log('Message Sent and Response received:', data);
//     document.getElementById('messageInput').value = '';
//   })
//   .catch(error => {
//     console.error('There was a problem with your fetch operation when sending message:', error);
//   });
// }