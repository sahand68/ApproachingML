function doGet(request) {

    var output  = ContentService.createTextOutput(),
      data    = {},
      id      = request.parameters.id,
      sheet   = request.parameters.sheet,
      ss      = SpreadsheetApp.openById(id);

  data.records = readData_(ss, sheet);
  
  var callback = request.parameters.callback;
  
  if (callback === undefined) {
    output.setContent(JSON.stringify(data));
  } else {
    output.setContent(callback + "(" + JSON.stringify(data) + ")");
  }
  output.setMimeType(ContentService.MimeType.JSON);
  
  return output; 
}


function readData_(ss, sheetname, properties) {

  if (typeof properties == "undefined") {
    properties = getHeaderRow_(ss, sheetname);
    //properties = properties.map(function(p) { return p.replace(/\s+/g, '_'); });
  }
  
  var rows = getDataRows_(ss, sheetname),
      data = [];

  for (var r = 0, l = rows.length; r < l; r++) {
    var row     = rows[r],
        record  = {};

    for (var p in properties) {
      record[properties[p]] = row[p];
    }
    
    data.push(record);

  }
  return data;
}


function getDataRows_(ss, sheetname) {
  var sh = ss.getSheetByName(sheetname);

  return sh.getRange(2, 1, sh.getLastRow() - 1, sh.getLastColumn()).getValues();
}


function getHeaderRow_(ss, sheetname) {
  var sh = ss.getSheetByName(sheetname);

  return sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0];  
}


//Recieve parameter and pass it to function to handle

function doPost(e) {
  //return handleResponse(e);
  //var sheet = ss.getSheetByName("Sheet1");
  var op       = e.parameter.action;
  var my_sheet = e.parameter.my_sheet;
  
  if(op=="add")
    return add_value(e,my_sheet);
  
  if(op=="update")
    return update_value(e,my_sheet);
  
  //return update_value(e);
}

// here handle with parameter

function add_value(request, my_sheet) {
  var output  = ContentService.createTextOutput();
  
  //create varibles to recieve respective parameters
    
  // // BARCODE,ID,EQUIPMENT_SCANNER,LOCATION, TIME_STAMP
  //var BARCODE       = request.parameter.BARCODE;
  var ID            = request.parameter.ID;
  var EQUIPMENT_SCANNER     = request.parameter.EQUIPMENT_SCANNER;
  var LOCATION      = request.parameter.LOCATION;
  var TIME_STAMP    = request.parameter.TIME_STAMP;


  var id = request.parameter.id;
  
  
  //open your Spread sheet by passing id
  
  var ss= SpreadsheetApp.openById(id);
  //var sheet=ss.getSheetByName("Sheet1");
  var sheet=ss.getSheetByName(my_sheet);
  
  //add new row with recieved parameter from client
  var rowData = sheet.appendRow([ID, EQUIPMENT_SCANNER, LOCATION, TIME_STAMP]);
 
  var callback = request.parameters.callback;
  if (callback === undefined) {
    output.setContent(JSON.stringify("Success"));
  } else {
    output.setContent(callback + "(" + JSON.stringify("Success") + ")");
  }

  output.setMimeType(ContentService.MimeType.JSON);
  
  return output;
}


//update function

function update_value(request,my_sheet) {
  
  var id = request.parameter.id;
  var ss = SpreadsheetApp.openById(id);
  //var sheet=ss.getSheetByName("Sheet1");
  var sheet=ss.getSheetByName(my_sheet);

var output             = ContentService.createTextOutput();
  //var qrcode_barcode   = request.parameter.qrcode_barcode;
  var QRCODE_BARCODE     = request.parameter.QRCODE_BARCODE;
  var flag=0;
  
  //var name = request.parameter.name;
  
  
  // // BARCODE,ID,EQUIPMENT_SCANNER,LOCATION, TIME_STAMP
  //var BARCODE       = request.parameter.BARCODE;
  var ID            = request.parameter.ID;
  var EQUIPMENT_SCANNER     = request.parameter.EQUIPMENT_SCANNER;
  var LOCATION      = request.parameter.LOCATION;
  var TIME_STAMP    = request.parameter.TIME_STAMP;    
  
  var lr= sheet.getLastRow();
  
  for(var i=1;i<=lr;i++){
    var rid = sheet.getRange(i, 1).getValue();
    if(rid==ID){
      //sheet.getRange(i,1).setValue(membership);
      
      // QRCODE_BARCODE, Product, Cost_CRC, Cost_USD, Target_Price_CRC, Target_Price_USD, Sold_Price_CRC, Sold_Price_USD, Comments
      sheet.getRange(i,2).setValue(EQUIPMENT_SCANNER);
      sheet.getRange(i,3).setValue(LOCATION);
      sheet.getRange(i,4).setValue(TIME_STAMP);

      
      var result="value updated successfully";
      flag=1;
    }
}
  if(flag==0)
    var result="id not found";
  
   result = JSON.stringify({
    "result": result
  });  
    
  return ContentService
  .createTextOutput(request.parameter.callback + "(" + result + ")")
  .setMimeType(ContentService.MimeType.JAVASCRIPT);   
   
}