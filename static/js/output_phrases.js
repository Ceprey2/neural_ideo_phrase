languages = ['ukrlang', 'ruslang', 'engllang', 'spanishlang', 'frenchlang', 'itallang', 'latinlang', 'hebrlang'];

function append_descriptors_to_select(select_id, descriptors_subdescriptors_json) {


    descriptors_subdescriptors_json.forEach(function (current_entry) {


        $('#' + select_id).append(`<option value=${current_entry["cluster"]}>
                                       ${current_entry["cluster"]}
                                  </option>`)

    });
}


function append_subdescriptors_to_select(select_id, subdescriptors_array) {

    $('#' + select_id).empty().append('<option selected="selected">Select subdescriptor</option>');
    subdescriptors_array.forEach(function (subdescriptors) {


        $('#' + select_id).append(`<option value=${subdescriptors}>
                                       ${subdescriptors}
                                  </option>`)

    });
}

append_descriptors_to_select("select_descriptor_k_means", json_descriptors_subdescriptors_k_means);
append_descriptors_to_select("select_descriptor_hierarchical", json_descriptors_subdescriptors_hierarchical);

let descriptor_k_means = $('#select_descriptor_k_means');
let descriptor_hierarchical = $('#select_descriptor_hierarchical');
let subdescriptor_hierarchical = $('#select_subdescriptors_hierarchical');

descriptor_k_means.change(function () {
    let subdescriptors = []
    json_descriptors_subdescriptors_k_means.forEach(function (entry) {
        if (descriptor_k_means.val().trim() == entry["cluster"])
            subdescriptors = entry["subclusters"].split(",");

    });
    append_subdescriptors_to_select("select_subdescriptor_k_means", subdescriptors);
    $('#div_ideas_k_means').html("Ideas related to " + descriptor_k_means.val().trim() + " <mark>" + subdescriptors + "</mark>");

});

descriptor_hierarchical.change(function () {
    let subdescriptors = []
    json_descriptors_subdescriptors_hierarchical.forEach(function (entry) {
        if (descriptor_hierarchical.val().trim() == entry["cluster"])
            subdescriptors = entry["subclusters"].split(",")
        //  alert(subdescriptors)

    });
    append_subdescriptors_to_select("select_subdescriptors_hierarchical", subdescriptors);
    $('#div_ideas_hierarchical').html("Ideas related to " + descriptor_hierarchical.val().trim() + " <mark>" + subdescriptors + "</mark>");
});


function get_phrases_according_descriptors(descriptor, subdescriptor, json_descriptors_subdescriptors, json_subdescriptors_phrases) {

    console.log("descriptor")
    console.log(descriptor)
    console.log("subdescriptor")
    console.log(subdescriptor)
    console.log("json_descriptors_subdescriptors: ")
    console.log(json_descriptors_subdescriptors[0])
    console.log("json_subdescriptors_phrases")
    console.log(json_subdescriptors_phrases[0])


    var phrases_str_to_return = [];

    json_descriptors_subdescriptors.forEach(function (element_descr_subd) {

        if (descriptor == element_descr_subd["cluster"].trim()) {

            languages.forEach(function (lang) {
                phrases_str_to_return.push(lang.toUpperCase() + ":" + "<br/>");

                json_subdescriptors_phrases.forEach(function (element_sub_phr) {

                    if (element_descr_subd["subclusters"].includes(element_sub_phr["subclusters"])) {
                        console.log("cluster includes subclusters and subdescriptor: " + subdescriptor)

                        if (subdescriptor == "Select subdescriptor" || subdescriptor == "") {



                                console.log("subdescriptor not chosen");
                                console.log("element_sub_phr - engllang");
                                console.log(element_sub_phr['engllang']);
                                if (typeof element_sub_phr[lang]==="undefined")  {
                                    console.log("undefined "+lang);
                                }
                                if (element_sub_phr[lang] != null && element_sub_phr[lang].trim()) { // mentioning object means, it has a value, it is not null neither empty string

                                    phrases_str_to_return.push(element_sub_phr[lang]+'#');
                                }


                        } else if (element_sub_phr["subclusters"] == subdescriptor) {

                            console.log("gggggg")
                            console.log(element_sub_phr[lang])

                            if (element_sub_phr[lang] != null && element_sub_phr[lang].trim()) { // mentioning object means, it has a value, it is not null neither empty string

                                phrases_str_to_return.push(element_sub_phr[lang]+'#');
                            }

                        }
                    }

                });

                phrases_str_to_return.push("<br/>");

            });
        }

    });
    console.log("phrases_str_to_return")
    console.log(phrases_str_to_return)
    return phrases_str_to_return;

}

descriptor_hierarchical.change(function () {

    console.log("66666$('#select_centroid').val()")
    console.log(descriptor_hierarchical.val())


    $('#div_phrases_hierarchical').html(get_phrases_according_descriptors(descriptor_hierarchical.val().trim(), subdescriptor_hierarchical.val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_hierarchical));
});

subdescriptor_hierarchical.change(function () {

    console.log("8888888$('#select_centroid').val()")
    console.log(subdescriptor_hierarchical.val())


    $('#div_phrases_hierarchical').html(get_phrases_according_descriptors(descriptor_hierarchical.val().trim(), subdescriptor_hierarchical.val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_hierarchical));
});


descriptor_k_means.change(function () {

    console.log("$('0000000#select_centroid').val()")
    console.log(descriptor_k_means.val())
    console.log(descriptor_k_means.text())


    $('#div_phrases_k_means').html(get_phrases_according_descriptors(descriptor_k_means.val().trim(), $('#select_subdescriptor_k_means').val().trim(), json_descriptors_subdescriptors_k_means, json_subdescriptors_phrases_k_means));
});

$('#select_subdescriptor_k_means').change(function () {

    console.log("eeeeeeee$('#select_centroid').val()")
    console.log($('#select_subdescriptor_k_means').val())


    $('#div_phrases_k_means').html(get_phrases_according_descriptors(descriptor_k_means.val().trim(), $('#select_subdescriptor_k_means').val().trim(), json_descriptors_subdescriptors_hierarchical, json_subdescriptors_phrases_k_means));  // TODO: create an output function
});

$('#btn_clear').click(function (){
    $('#div_phrases_k_means').text("");
    $('#div_ideas_hierarchical').text("");
    $('#div_phrases_hierarchical').text("");
    $('#div_ideas_k_means').text("");
});
