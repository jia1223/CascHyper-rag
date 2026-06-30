export const storage = function (key, value) {
    localStorage.setItem(key, JSON.stringify(value))
    return value
}

export const SERVER_URL = import.meta.env.VITE_SERVER_URL || 'http://localhost:8000';