$(document).ready(function () {
  // var socket = io();
  // var socket = io.connect('https://reknowdesktopsurveillance.hiit.fi:8080');

  var selectedEntities = [];
  var people, applications, documents, topics;
  var dataKeys;
  var htmlListName = ["topicList", "applicationList", "documentList",  "peopleList"];
  var newData, oldData;
  var enterData, updateData, exitData;
  // socket.on('sendAllLists', function(data){
  // socket.on('entityData', function(data){
  setInterval(function() {
    $("#sysStatus").load("/retrieve", function(data) {
      oldData = newData;
      newData = JSON.parse(data);
      enterData = [], updateData = [], exitData = [];
      console.log('get all lists data from backend', newData);

      people = newData.people;
      applications = newData.applications;
      documents = newData.document_ID;
      topics = newData.keywords;
      dataKeys = Object.keys(newData);
      dataKeys.splice(dataKeys.indexOf("pair_similarity"), 1);

      // create enterData and updateData
      for (var i = 0; i < dataKeys.length; i++) {
        for (var j = 0; j < newData[dataKeys[i]].length; j++) {
          var currentId = newData[dataKeys[i]][j][0];
          var currentVisualEntity = { htmlListName:htmlListName[i], column:j, id:currentId.toString(), label:newData[dataKeys[i]][j][1]};
          if (!idInOldData(currentId)) {
            enterData.push(currentVisualEntity);
          } else {
            updateData.push(currentVisualEntity);
          }
        }
      }

      // create exitData
      for (var i = 0; i < dataKeys.length; i++) {
        if (typeof oldData == 'undefined') {
          break;
        }
        for (var j = 0; j < oldData[dataKeys[i]].length; j++) {
          var currentId = oldData[dataKeys[i]][j][0];
          var currentVisualEntity = { htmlListName:htmlListName[i], column:j, id:currentId.toString(), label:oldData[dataKeys[i]][j][1]};
          if (!idInNewData(currentId)) {
            exitData.push(currentVisualEntity);
          }
        }
      }

      function idInOldData(id) {
        var oldDataId;
        if (typeof oldData == 'undefined') {
          return false;
        }
        for (var i = 0; i < dataKeys.length; i++) {
          for (var j = 0; j < oldData[dataKeys[i]].length; j++) {
            oldDataId = oldData[dataKeys[i]][j][0];
            if (oldDataId == id) {
              return true;
            }
          }
        }
        return false;
      }

      function idInNewData(id) {
        var newDataId;
        for (var i = 0; i < dataKeys.length; i++) {
          for (var j = 0; j < newData[dataKeys[i]].length; j++) {
            newDataId = newData[dataKeys[i]][j][0];
            if (newDataId == id) {
              return true;
            }
          }
        }
        return false;
      }

      // for enterData, add the div
      for (var i = 0; i < enterData.length; i++) {
        $( "#"+enterData[i].htmlListName )
          .append($('<div></div>')
            .addClass("entity")
            // .css({"left":enterData[i].column * 270 + "px"})
            .css({"left": "1350px"})
            .animate({"left":enterData[i].column * 270 + "px"}, 1000)
            .click(function(ev) {
              var label, id;
              var htmlListName = $(ev.target).parents(".list").attr("id");
              var selectStatusWhenClicking = null;
              // toggle class/bg
              if ($(ev.target).hasClass("entity")) {
                label = $(ev.target).find(".label").text();
                id = $(ev.target).find(".profile").attr("class").match(/id[\w-]*\b/)[0].substr(2);
                $(ev.target).toggleClass("selected");
                if ($(ev.target).hasClass("selected")) {
                  selectStatusWhenClicking = false;
                } else {
                  selectStatusWhenClicking = true;
                }
              } else {
                label = $(ev.target).parents(".entity").find(".label").text();
                id = $(ev.target).parents(".entity").find(".profile").attr("class").match(/id[\w-]*\b/)[0].substr(2);
                $(ev.target).parents(".entity").toggleClass("selected");
                if ($(ev.target).parents(".entity").hasClass("selected")) {
                  selectStatusWhenClicking = false;
                } else {
                  selectStatusWhenClicking = true;
                }
              }
              // toggle selection
              toggleSelection({ htmlListName:htmlListName, id:id.toString(), label:label});

              if (selectStatusWhenClicking) {
                // socket.emit('userFeedback', JSON.stringify({"user_feedback":[[id, 0]]}));
                console.log(JSON.stringify({"user_feedback":[[id, 0]]}));
              } else {
                // socket.emit('userFeedback', JSON.stringify({"user_feedback":[[id, 1]]}));
                // if (id == "7") {
                //   socket.emit('send1AfterSkpyeData');
                // } else if (id == "388") {
                //   socket.emit('send2AfterFBData');
                // } else if (id == "3049") {
                //   socket.emit('send3AfterTuukkaData');
                // } else if (id == "2699") {
                //   socket.emit('send2AfterFBData');
                // }
                console.log(JSON.stringify({"user_feedback":[[id, 1]]}));
              }
            })
            .append($('<div></div>')
              .addClass("profile id" + enterData[i].id))
            .append($('<div></div>')
              .addClass("label")));

        var displayedLabel = enterData[i].label;
        var subDisplayedLabel;
        var lastIndexOfSpace;

        // for applications, replace "_" with "."
        if (enterData[i].htmlListName == "applicationList") {
          displayedLabel = displayedLabel.replace(/_/g , ".");
        }

        // for document title, put to 2 lines and trim it if too long
        if (enterData[i].htmlListName == "documentList") {
          subDisplayedLabel = displayedLabel.substr(0, 31);
          lastIndexOfSpace = subDisplayedLabel.lastIndexOf(" ");
          // for the first line
          //  if the first line does not have space, we break it after 32th char
          if (lastIndexOfSpace == -1) {
            displayedLabel = displayedLabel.slice(0, 32) + "\n" + displayedLabel.slice(32);
            // for the second line
            if (displayedLabel.length > 65) {
              displayedLabel = displayedLabel.slice(0, 64) + "…";
            }
          }
          //  if the first line has space, we break it after the last space
          else {
            displayedLabel = displayedLabel.slice(0, lastIndexOfSpace) + "\n" + displayedLabel.slice(lastIndexOfSpace + 1);
            // for the second line
            if (displayedLabel.length - lastIndexOfSpace - 1 > 32) {
              displayedLabel = displayedLabel.slice(0, lastIndexOfSpace + 1 + 31) + "…";
            }
          }
        }

        $('#' + enterData[i].htmlListName + ' .id' + enterData[i].id).parents(".entity").find(".label").text(displayedLabel);
      }

      // for updateData, change the position
      for (var i = 0; i < updateData.length; i++) {
        $(' .id' + updateData[i].id).parents(".entity").animate({"left":updateData[i].column * 270 + "px"}, 1000);
      }

      // for exitData, change the position
      for (var i = 0; i < exitData.length; i++) {
        if (exitData.length == 0) {
          break;
        }
        var $entity = $('#flowingArea .id' + exitData[i].id).parents(".entity");
        $entity.fadeOut(1000);
        setTimeout(function(){
          $entity.remove();
        }, 1000)
      }
    });
  },10000); // call every 10 seconds

  function toggleSelection(entity) {
    if (selectedEntities.map(function(a) {return a.id;}).indexOf(entity.id) == -1) {
      selectedEntities.push(entity);
    } else {
      selectedEntities.splice(selectedEntities.map(function(a) {return a.id;}).indexOf(entity.id), 1);
    }
    updateSecreenSeletedList();
  }

  function updateSecreenSeletedList() {
    if (selectedEntities.length == 0) {
      $("#selection").text("Select items to interact with");
      $("#selection").addClass("empty");
    } else {
      $("#selection").text("");
      $("#selection").removeClass("empty");
      $("#selection").find(".entity").remove();

      for (var i = 0; i < selectedEntities.length; i++) {
        $("#selection")
          .append($('<div></div>')
            .addClass("entity " + selectedEntities[i].htmlListName)
            .append($('<div></div>')
              .addClass("profile id" + selectedEntities[i].id))
            .append($('<div></div>')
              .addClass("label")
              .text(selectedEntities[i].label.length > 20 ? selectedEntities[i].label.slice(0, 19) + "…" : selectedEntities[i].label))
            .append($('<div>x</div>')
              .addClass("delete")
              .click(function(ev) {
                var id;
                if ($(ev.target).hasClass("entity")) {
                  id = $(ev.target).find(".profile").parents(".entity").attr("class").match(/id[\w-]*\b/)[0].substr(2);
                } else {
                  id = $(ev.target).parents(".entity").find(".profile").attr("class").match(/id[\w-]*\b/)[0].substr(2);
                }

                //update front-end
                selectedEntities.splice(selectedEntities.map(function(a) {return a.id;}).indexOf(id), 1);
                $("#flowingArea").find(".id" + id).parents(".entity").toggleClass("selected");
                updateSecreenSeletedList();

                //update back-end
                // socket.emit('userFeedback', JSON.stringify({"user_feedback":[[id, 0]]}));
                console.log(JSON.stringify({"user_feedback":[[id, 0]]}));
              })));
      }
    }
  }

  $( "#clearSelection" ).click(function(){
    $('.entity').removeClass("selected");
    var id;
    for (var i = (selectedEntities.length - 1); i >= 0; i--) {
      id = selectedEntities[i].id;
      // socket.emit('userFeedback', JSON.stringify({"user_feedback":[[id, 0]]}));
      console.log(JSON.stringify({"user_feedback":[[id, 0]]}));
      selectedEntities.splice( i, 1);
    }
    updateSecreenSeletedList();
  });

  // $( "#sendInitialDataButton" ).click(function(){
  //   socket.emit('sendInitialData');
  // });
  // $( "#send1AfterSkpyeDataButton" ).click(function(){
  //   socket.emit('send1AfterSkpyeData');
  // });
  // $( "#send2AfterFBDataButton" ).click(function(){
  //   socket.emit('send2AfterFBData');
  // });
  // $( "#send3AfterTuukkaDataButton" ).click(function(){
  //   socket.emit('send3AfterTuukkaData');
  // });
});
