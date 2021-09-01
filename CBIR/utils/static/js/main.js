// ----- custom js ----- //

var items = [];
var items_for_reranking = [];
var query_images = [];
var $cell = $('.image__cell');

$("#results-table").hide();
$("#error").hide();
$("#reranking-button").hide();
$("#div__query").hide();


$(function () {

    // sanity check
    console.log("ready!");

    /* ------------------------------
     検索ボタンを押したときの動作
     #search-form: 検索ボタン
    ------------------------------ */
    $('#search-form').submit(function (event) {
        $("#keyword-search-results").empty();
        $("#reranking-button").hide();
        $("#results").empty();
        $("#pagination").empty();
        $("#pagination-main").twbsPagination('destroy');
        $("#div__query").empty();
        $("#div__query").hide();
        query_images = [];
        // HTMLでの送信をキャンセル
        event.preventDefault();
        var $form = $(this);
        var $keyword = $('#keyword').val();
        var $algorithm = $('#algorithm-select').val();


        if ($keyword == "") {
            alert('キーワードを入力してください．');
            event.preventDefault();
            return false;
        }

        $("#results-table").hide();
        $("#error").hide();

        dispLoading("読込中...");

        var $pagination_main = $('#pagination-main'),
        totalRecords_main = 0,
        records_main = [],
        displayRecords_main = [],
        recPerPage_main = 60,
        page_main = 1,
        totalPages_main = 0;

        $.ajax({
            url: '/',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({'keyword': $keyword, 'algorithm': $algorithm}),
            success: function (result) {
                var data = result.results;
                data = JSON.parse(data);

                totalRecords_main = Object.keys(data).length;

                if(totalRecords_main == 0) {
                    $('#keyword-search-results').html('お探しの商品は存在しません．もう一度キーワードを入力してください．');
                    return;
                };

                records_main = Object.entries(data);
                totalPages_main = Math.ceil(totalRecords_main / recPerPage_main);

                apply_pagination();

            },
            // handle error
            error: function (error) {
                removeLoading();
                console.log(error);
                // append to dom
                $("#error").append()
            },
            complete: function(){
                removeLoading();
                $("#div__query").show();
            },
        });

        function apply_pagination() {
            $pagination_main.twbsPagination({
                totalPages: totalPages_main,
                visiblePages: 6,
                first: '最初',
                prev : '前',
                next : '次',
                last : '最後',
                onPageClick: function (event, page) {
                    displayRecordsIndex_main = Math.max(page - 1, 0) * recPerPage_main;
                    endRec_main = (displayRecordsIndex_main) + recPerPage_main;

                    displayRecords_main = records_main.slice(displayRecordsIndex_main, endRec_main);
                    generate_table();
                }
            });
        }

        function generate_table() {
              var tr='';
              $('#keyword-search-results').html('');
              $("#keyword-search-results").append('<h3>「'+$keyword+'」の検索結果（'+totalRecords_main+'件）</h3>');
              $("#keyword-search-results").append(
                        '<div class="row">' +
                            '<div class="col text-center">' +
                                '<div class="col-xs-2">' +
                                    '<button type="submit" class="btn btn-warning" name="search" id="search"' +
                                    'value="類似商品画像検索">類似商品画像検索' +
                                    '</button>' +
                                '</div>' +
                                '<div class="col-xs-8">' +
                                    '<p class="help-block" >右上のチェックマークを選択した商品画像と類似する商品を検索します．</p>'+
                                '</div>' +
                            '</div>' +
                        '</div>');
              for (var i = 0; i < displayRecords_main.length; i++) {
                    tr+= '<div class="image__cell is-collapsed">' +
                                '<div>' +
                                    '<input type="checkbox" id="like--'+displayRecords_main[i][0]+'">'+
                                    '<label for="like--'+displayRecords_main[i][0]+'">' +
                                    '<div class="check__box" id="'+displayRecords_main[i][0]+'"></div>'+
                                    '</label>' +
                                '</div>' +
                                '<div class="image--basic">'+
                                    '<a href="#'+displayRecords_main[i][0]+'"><img class="basic__img" id=image--'+displayRecords_main[i][0]+
                                    ' src="static/img/'+displayRecords_main[i][1]['top_img']+'">' +
                                    '</a>'+
                                    '<figcaption>'+displayRecords_main[i][1]['name']+'</figcaption>'+
                                    '<div class="arrow--up"></div>' +
                                '</div>';
                    tr+= '<div class="image--expand">'+
                                '<a href="#'+displayRecords_main[i][0]+'" class="expand__close"></a>'+
                                '<div class="row">'+
                                    '<div class="col-xs-6">'+
                                        '<img class="image--large" src="static/img/'+displayRecords_main[i][1]['top_img']+'">'+
                                    '</div>'+
                                    '<div class="col-xs-4">'+
                                        '<h3>機械詳細</h3>'+
                                        '<table class="table" cellspacing="0">'+
                                            '<thead>'+
                                            '<tr>'+
                                                '<th></th>'+
                                                '<th></th>'+
                                            '</tr>'+
                                            '</thead>'+
                                            '<tbody>'+
                                            '<tr>'+
                                                '<th scope="row">出品会社</th>'+
                                                // '<td>出品会社</td>'+
                                                '<td></td>'+
                                            '</tr>'+
                                            '<tr>'+
                                                '<th scope="row">管理番号</th>'+
                                                // '<td>管理番号</td>'+
                                                '<td>'+displayRecords_main[i][0]+'</td>'+
                                            '</tr>'+
                                            '<tr>'+
                                                '<th scope="row">ジャンル</th>'+
                                                // '<td>ジャンル</td>'+
                                                '<td>'+displayRecords_main[i][1]['genre_id']+'</td>'+
                                            '</tr>'+
                                            '<tr>'+
                                                '<th scope="row">機械名</th>'+
                                                // '<td>機械名</td>'+
                                                '<td>'+displayRecords_main[i][1]['name']+'</td>'+
                                            '</tr>'+
                                            '<tr>'+
                                                '<th scope="row">メーカー</th>'+
                                                // '<td>メーカー</td>'+
                                                '<td>'+displayRecords_main[i][1]['maker']+'</td>'+
                                            '</tr>'+
                                            '<tr>'+
                                                '<th scope="row">モデル</th>'+
                                                // '<td>モデル</td>'+
                                                '<td>'+displayRecords_main[i][1]['model']+'</td>'+
                                            '</tr>'+
                                            '<tr>'+
                                                '<th scope="row">スペック</th>'+
                                                // '<td>スペック</td>'+
                                                '<td style="word-wrap: break-word">'+displayRecords_main[i][1]['spec']+'</td>'+
                                            '</tr>'+
                                            '<tr>'+
                                                '<th scope="row">一山</th>'+
                                                // '<td>スペック</td>'+
                                                '<td style="word-wrap: break-word">'+displayRecords_main[i][1]['hitoyama']+'</td>'+
                                            '</tr>'+
                                            '</tbody>'+
                                        '</table>'+
                                   '</div>'+
                                '</div>'+
                            '</div>';
                    tr+='</div>';
              }
              $("#keyword-search-results").append('<section class="image-grid">'+tr+'</section>');
        }

    });

    /* ------------------------------
     類似商品画像検索ボタンを押したときの動作
     search: 検索ボタン
    ------------------------------ */
    $(document).on('click',"[ name = 'search' ]", function(event) {
        console.log(items)
        event.preventDefault();
        if (items.length == 0) {
            alert('所望の商品にチェックを入れてください．');
            event.preventDefault();
            return false;
        }

        $("#results").empty();
        $("#pagination").twbsPagination('destroy');
        $("#reranking-button").show();
        $("#error").hide();

        console.log($("#pagination"))

        var $pagination = $('#pagination'),
        totalRecords = 0,
        records = [],
        displayRecords = [],
        recPerPage = 20,
        page = 1,
        totalPages = 0;

        $.ajax({
            url: '/search',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({"ids": items}),
            success: function (result) {
                // console.log(result.results);
                var data = result.results;

                records = data;
                totalRecords = records.length;
                totalPages = Math.ceil(totalRecords / recPerPage);

                $("#results-table").show();

                apply_pagination();

            },
            // handle error
            error: function (error) {
                // console.log(error);
                // append to dom
                $("#error").append()
            }
        });

        function apply_pagination() {
            $pagination.twbsPagination({
                totalPages: totalPages,
                visiblePages: 3,
                first: '最初',
                prev : '前',
                next : '次',
                last : '最後',
                onPageClick: function (event, page) {
                    displayRecordsIndex = Math.max(page - 1, 0) * recPerPage;
                    endRec = (displayRecordsIndex) + recPerPage;

                    displayRecords = records.slice(displayRecordsIndex, endRec);
                    generate_table();
                }
            });
        }

        function generate_table() {
              var tr;
              $('#results').html('');
              for (var i = 0; i < displayRecords.length; i++) {
                    tr = $('<tr/>');
                    tr.append('<th><a href="' + displayRecords[i]["image"] + '"><img src="' + displayRecords[i]["image"] + '" class="result-img"></a></th>');
                    tr.append("<th>" + displayRecords[i]['score'] + "</th>");
                    $('#results').append(tr);
              }
              $('#results').append('<p>候補数（'+totalRecords+'件）</p>');
        }

        $('div').removeClass("active");
        $( 'input[type="checkbox"]' ).prop('checked', false);

        items_for_reranking = items.concat();
        items = [];
    });

    $("[ name = 'reranking' ]").on('click', function () {

        $("#results").empty();
        $("#pagination").twbsPagination('destroy');

        var $pagination = $('#pagination'),
        totalRecords = 0,
        records = [],
        displayRecords = [],
        recPerPage = 20,
        page = 1,
        totalPages = 0;

        $.ajax({
            url: '/reranking',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({"ids": items_for_reranking}),
            success: function (result) {
                // console.log(result.results);
                var data = result.results;

                records = data;
                totalRecords = records.length;
                totalPages = Math.ceil(totalRecords / recPerPage);

                $("#results-table").show();

                apply_pagination();

            },
            // handle error
            error: function (error) {
                // console.log(error);
                // append to dom
                $("#error").append()
            }
        });

        function apply_pagination() {
            $pagination.twbsPagination({
                totalPages: totalPages,
                visiblePages: 3,
                first: '最初',
                prev : '前',
                next : '次',
                last : '最後',
                onPageClick: function (event, page) {
                    displayRecordsIndex = Math.max(page - 1, 0) * recPerPage;
                    endRec = (displayRecordsIndex) + recPerPage;

                    displayRecords = records.slice(displayRecordsIndex, endRec);
                    generate_table();
                }
            });
        }

        function generate_table() {
              var tr;
              $('#results').html('');
              for (var i = 0; i < displayRecords.length; i++) {
                    tr = $('<tr/>');
                    tr.append('<th><a href="' + displayRecords[i]["image"] + '"><img src="' + displayRecords[i]["image"] + '" class="result-img"></a></th>');
                    tr.append("<th>" + displayRecords[i]['score'] + "</th>");
                    $('#results').append(tr);
              }
              $('#results').append('<p>候補数（'+totalRecords+'件）</p>');
        }

        $('div').removeClass("active");
        $( 'input[type="checkbox"]' ).prop('checked', false);
    });

    /* ------------------------------
     メインフィールドの画像を選択したときの動作
    ------------------------------ */
    $(document).on('click','.image__cell .image--basic', function() {
        var $thisCell = $(this).closest('.image__cell');
        if ($thisCell.hasClass('is-collapsed')) {
            $('.image__cell').not($thisCell).removeClass('is-expanded').addClass('is-collapsed');
            $thisCell.removeClass('is-collapsed').addClass('is-expanded');
        } else {
            $thisCell.removeClass('is-expanded').addClass('is-collapsed');
        }
    });

    /* ------------------------------
     機械詳細を閉じる
    ------------------------------ */
    $(document).on('click','.image__cell .expand__close', function() {
        var $thisCell = $(this).closest('.image__cell');
        $thisCell.removeClass('is-expanded').addClass('is-collapsed');
    });

    /* ------------------------------
     画像にチェックマークをつける
    ------------------------------ */
    $(document).on('click','.image__cell .check__box', function() {

        if( !$('#results').is(':empty') && items.length == 0) {
            query_images = [];
        }
        // $("#results-table").hide();
        $("#error").hide();

        var image = $(this).attr("id");
        var path = $('#image--'+image).attr("src");
        var html = '<img class=image__query src='+path+'>'
        //$("#div__query").append('<img class=image__cell src='+path+'>');

        console.log(path);

        $("#div__query").html('');
        if ($(this).hasClass("active")) {
            $(this).removeClass("active");
            // remove a item from a list
            query_images = query_images.filter(n => n !== html);
            items = items.filter(n => n !== image);
        } else {
            $(this).addClass("active");
            items.push(image)
            query_images.push(html)
        }

       for(var i=0; i<query_images.length; i++) {
            $("#div__query").append(query_images[i]);
       }

        console.log(items)
    });

});

/* ------------------------------
 Loading イメージ表示関数
 引数： msg 画面に表示する文言
 ------------------------------ */
function dispLoading(msg){
  // 引数なし（メッセージなし）を許容
  if( msg == undefined ){
    msg = "";
  }
  // 画面表示メッセージ
  var dispMsg = "<div class='loadingMsg'>" + msg + "</div>";
  // ローディング画像が表示されていない場合のみ出力
  if($("#loading").length == 0){
    $("body").append("<div id='loading'>" + dispMsg + "</div>");
  }
}

/* ------------------------------
 Loading イメージ削除関数
 ------------------------------ */
function removeLoading(){
  $("#loading").remove();
}


