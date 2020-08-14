//  var centroids_arr = centroids.split(" ");

console.log("1")
console.log(json_subdescriptors_phrases_k_means[0]['subclusters'])
console.log("2")
console.log(json_descriptors_subdescriptors_k_means[0]['clusters'])
console.log("3")
console.log(json_subdescriptors_phrases_hierarchical[0]['subclusters'])
console.log("4")
console.log(json_descriptors_subdescriptors_hierarchical[0]['clusters'])
console.log(json_descriptors_subdescriptors_hierarchical[1]['clusters'])
console.log(json_descriptors_subdescriptors_hierarchical[2]['clusters'])
console.log(json_descriptors_subdescriptors_hierarchical[3]['clusters'])
console.log(json_descriptors_subdescriptors_hierarchical[4]['clusters'])
//        console.log("first element from dict_centroid_phrases|safe")
//        console.log(var_centroid_phrases[0])
//
//        var parsed_centroid_phrases = JSON.parse(var_centroid_phrases[0]);
//          console.log("main_centroid of the first data element")
//          console.log(parsed_centroid_phrases["main_centroid"])
//        console.log("centroids centroids_arr")
//
//      function get_main_centroids_from_dict (dict){
//
//              var descriptors_set = new Set();
//              var subdescriprotrs_set = new Set();
//
//
//              dict.forEach(function (element){
//
//                  //alert(element);
//
//                  current_entry = JSON.parse(element);
//
//                  current_entry['entries'].forEach(function (subelement){
//
//                      current_subentry = JSON.parse(subelement);
//
//                       subdescriprotrs_set.add(current_subentry["centroid"]);
//                      // alert("subcentroid " + current_subentry["centroid"])
//                  });
//
//                descriptors_set.add(current_entry["main_centroid"])
//              });
//
//              return [descriptors_set, subdescriprotrs_set];
//        };
//
//
//
//          var centroids_subcentroids_k_means = get_main_centroids_from_dict(var_centroid_phrases);
//
//          alert(centroids_subcentroids_k_means);
//          alert((dict_centroid_phrases_hierarchical[1]["phrases"]));
//
//
//
// var centroids_subcentroids_hierarchical = get_main_centroids_from_dict(dict_centroid_phrases_hierarchical);
//
//          alert(centroids_subcentroids_hierarchical);

function append_descriptors_to_select(select_id, descriptors_subdescriptors_json) {


    descriptors_subdescriptors_json.forEach(function (current_entry) {


        $('#' + select_id).append(`<option value=${current_entry["clusters"]}>
                                       ${current_entry["clusters"]}
                                  </option>`)

    });
}


function append_subdescriptors_to_select(select_id, subdescriptors_array) {

    $('#' + select_id).empty().append('<option selected="selected" value="subdescriptor">subdescriptor</option>')
    ;
    subdescriptors_array.forEach(function (subdescriptors) {


        $('#' + select_id).append(`<option value=${subdescriptors}>
                                       ${subdescriptors}
                                  </option>`)

    });
}

append_descriptors_to_select("select_descriptor_k_means", json_descriptors_subdescriptors_k_means);
append_descriptors_to_select("select_descriptor_hierarchical", json_descriptors_subdescriptors_hierarchical);

$('#select_descriptor_k_means').change(function () {
    subdescriptors = []
    json_descriptors_subdescriptors_k_means.forEach(function (entry) {
        if ($('#select_descriptor_k_means').val().trim() == entry['clusters'])
            subdescriptors = entry["subclusters"].split(", ");
        // alert(subdescriptors)


    });
    append_subdescriptors_to_select("select_subdescriptor_k_means", subdescriptors);

});

$('#select_descriptor_hierarchical').change(function () {
    subdescriptors = []
    json_descriptors_subdescriptors_hierarchical.forEach(function (entry) {
        if ($('#select_descriptor_hierarchical').val().trim() == entry['clusters'])
            subdescriptors = entry["subclusters"].split(", ")
        //  alert(subdescriptors)

    });
    append_subdescriptors_to_select("select_subdescriptors_hierarchical", subdescriptors);
});

// append_descriptors_to_select ("select_centroid_hierarchical", centroids_subcentroids_hierarchical[0]);
// append_descriptors_to_select ("select_centroid_hierarchical_level2", centroids_subcentroids_hierarchical[1]);


function get_phrases_according_descriptors(descriptor, subdescriptor, json_descriptors_subdescriptors, json_subdescriptors_phrases) {

    var phrases_str_to_return = [];

    json_descriptors_subdescriptors.forEach(function (element_descr_subd) {

        // alert('%'+parsed_element["centroid"].trim()+'%'+ " "+'%'+$('#select_centroid').val()+'%');
        // alert('%'+parsed_element["centroid"].trim()+'%');
        if (descriptor == element_descr_subd["clusters"].trim()) {
                alert(element_descr_subd["subclusters"]+" "+subdescriptor)



                json_subdescriptors_phrases.forEach(function (element_sub_phr) {

                     if (element_descr_subd["subclusters"].includes(element_sub_phr["subclusters"])) {
                         // if is found an entry with a subclustar as those indicated in the respective entry of element_descr_subd

                         alert("includes "+element_sub_phr["subclusters"]);

                         // if (element_sub_phr['phrases'] != null && element_sub_phr["subcluster"] == subdescriptor) {
                         //     phrases_str_to_return.push(element_sub_phr['phrases']);
                         //
                         // } else
                         //
                         //     {

                            if ( subdescriptor == "subdescriptor" || subdescriptor == "") {
    if (element_sub_phr['phrases'] != null) phrases_str_to_return.push(element_sub_phr['phrases']);

}
                            else if (element_sub_phr['phrases'] != null && element_sub_phr["subclusters"] == subdescriptor){

                                phrases_str_to_return.push(element_sub_phr['phrases']);
                            }

                       //  }
                     }

                });

        }

    });
return phrases_str_to_return;

}

$('#select_descriptor_hierarchical').change(function () {

    console.log("$('#select_centroid').val()")
    console.log($('#select_descriptor_hierarchical').val())

    $('#div_ideas').text("ideas");
    $('#div_phrases').text(get_phrases_according_descriptors($('#select_descriptor_hierarchical').val().trim(), $('#select_subdescriptors_hierarchical').val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_hierarchical));
});

$('#select_subdescriptors_hierarchical').change(function () {

    console.log("$('#select_centroid').val()")
    console.log($('#select_subdescriptors_hierarchical').val())

    $('#div_ideas').text("ideas");
    $('#div_phrases').text(get_phrases_according_descriptors($('#select_descriptor_hierarchical').val().trim(), $('#select_subdescriptors_hierarchical').val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_hierarchical));
});


$('#select_descriptor_k_means').change(function () {

    console.log("$('#select_centroid').val()")
    console.log($('#select_descriptor_k_means').val())

    $('#div_ideas').text("ideas");
    $('#div_phrases').text(get_phrases_according_descriptors($('#select_descriptor_k_means').val().trim(), $('#select_subdescriptor_k_means').val().trim(), json_descriptors_subdescriptors_k_means, json_subdescriptors_phrases_k_means));
});

$('#select_subdescriptor_k_means').change(function () {

    console.log("$('#select_centroid').val()")
    console.log($('#select_subdescriptor_k_means').val())

    $('#div_ideas').text("ideas");
    $('#div_phrases').text(get_phrases_according_descriptors($('#select_descriptor_k_means').val().trim(), $('#select_subdescriptor_k_means').val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_k_means));
});


