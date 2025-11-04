struct Polar3
    pyobj::PyObject
end

"read_polar3: python Polar3 객체를 Polar3 struct에 래핑"
function read_polar3(filename; path="", kwargs...)
    fpath = path == "" ? filename : joinpath(path, filename)
    airfoilprep = pyimport("airfoilprep")
    return Polar3(airfoilprep[:read_polar3](fpath))
end

# getter 함수 (output 형식 맞춤)
function get_cl(polar::Polar3, Re, Alpha)
    return pycall(polar.pyobj[:get_cl], Float64, Re, Alpha)
end
function get_cd(polar::Polar3, Re, Alpha)
    return pycall(polar.pyobj[:get_cd], Float64, Re, Alpha)
end
function get_cm(polar::Polar3, Re, Alpha)
    return pycall(polar.pyobj[:get_cm], Float64, Re, Alpha)
end
