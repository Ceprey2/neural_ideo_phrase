


function append_descriptors_to_select(select_id, descriptors_subdescriptors_json) {


    descriptors_subdescriptors_json.forEach(function (current_entry) {


        $('#' + select_id).append(`<option value=${current_entry["clusters"]}>
                                       ${current_entry["clusters"]}
                                  </option>`)

    });
}


function append_subdescriptors_to_select(select_id, subdescriptors_array) {

    $('#' + select_id).empty().append('<option selected="selected" value="subdescriptor">Select idea</option>')
    ;
    subdescriptors_array.forEach(function (subdescriptors) {


        $('#' + select_id).append(`<option value=${subdescriptors}>
                                       ${subdescriptors}
                                  </option>`)

    });
}

append_descriptors_to_select("select_descriptor_k_means", json_descriptors_subdescriptors_k_means);
append_descriptors_to_select("select_descriptor_hierarchical", json_descriptors_subdescriptors_hierarchical);

const descriptor_k_means = $('#select_descriptor_k_means');
const descriptor_hierarchical = $('#select_descriptor_hierarchical');

descriptor_k_means.change(function () {
    subdescriptors = []
    json_descriptors_subdescriptors_k_means.forEach(function (entry) {
        if (descriptor_k_means.val().trim() == entry['clusters'])
            subdescriptors = entry["subclusters"].split(", ");
      
    });
    append_subdescriptors_to_select("select_subdescriptor_k_means", subdescriptors);

});

descriptor_hierarchical.change(function () {
    subdescriptors = []
    json_descriptors_subdescriptors_hierarchical.forEach(function (entry) {
        if (descriptor_hierarchical.val().trim() == entry['clusters'])
            subdescriptors = entry["subclusters"].split(", ")
        //  alert(subdescriptors)

    });
    append_subdescriptors_to_select("select_subdescriptors_hierarchical", subdescriptors);
});



function get_phrases_according_descriptors(descriptor, subdescriptor, json_descriptors_subdescriptors, json_subdescriptors_phrases) {

    var phrases_str_to_return = [];

    json_descriptors_subdescriptors.forEach(function (element_descr_subd) {

        if (descriptor == element_descr_subd["clusters"].trim()) {

                json_subdescriptors_phrases.forEach(function (element_sub_phr) {

                     if (element_descr_subd["subclusters"].includes(element_sub_phr["subclusters"])) {


                            if ( subdescriptor === "Select idea" || subdescriptor === "") {
    if (element_sub_phr["ukrlang"] != null) phrases_str_to_return.push(element_sub_phr["ukrlang"]);

}
                            else if (element_sub_phr["ukrlang"] != null && element_sub_phr["subclusters"] == subdescriptor){

                                phrases_str_to_return.push(element_sub_phr["ukrlang"]);
                            }
                     }

                });
        }

    });
return phrases_str_to_return;

}

descriptor_hierarchical.change(function () {

    console.log("$('#select_centroid').val()")
    console.log(descriptor_hierarchical.val())

    $('#div_ideas').text("ideas");
    $('#div_phrases_hierarchical').text(get_phrases_according_descriptors(descriptor_hierarchical.val().trim(), $('#select_subdescriptors_hierarchical').val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_hierarchical));
});

$('#select_subdescriptors_hierarchical').change(function () {

    console.log("$('#select_centroid').val()")
    console.log($('#select_subdescriptors_hierarchical').val())

    $('#div_ideas').text("ideas");
    $('#div_phrases_hierarchical').text(get_phrases_according_descriptors(descriptor_hierarchical.val().trim(), $('#select_subdescriptors_hierarchical').val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_hierarchical));
});


descriptor_k_means.change(function () {

    console.log("$('#select_centroid').val()")
    console.log(descriptor_k_means.val())

    $('#div_ideas').text("ideas");
    $('#div_phrases').text(get_phrases_according_descriptors(descriptor_k_means.val().trim(), $('#select_subdescriptor_k_means').val().trim(), json_descriptors_subdescriptors_k_means, json_subdescriptors_phrases_k_means));
});

$('#select_subdescriptor_k_means').change(function () {

    console.log("$('#select_centroid').val()")
    console.log($('#select_subdescriptor_k_means').val())

    $('#div_ideas').text("ideas");
    $('#div_phrases').text(get_phrases_according_descriptors(descriptor_k_means.val().trim(), $('#select_subdescriptor_k_means').val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_k_means));  // TODO: create an output function
});


